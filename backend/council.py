"""Google ADK-backed 3-stage council orchestration."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import uuid
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from urllib.parse import urlparse

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    from google.adk.tools.google_search_tool import GoogleSearchTool
    from google.adk.utils.model_name_utils import is_gemini_model
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.models.lite_llm import LiteLlm
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - surfaced at runtime for clarity
    raise RuntimeError(
        "Google ADK dependencies are missing. "
        "Install `google-adk` and `google-genai` to run the council."
    ) from exc

from .config import (
    ADK_ALLOWED_FILE_ROOT,
    ADK_CHAIRMAN_MODEL,
    ADK_COUNCIL_MEMBERS,
    ADK_FILE_TOOL_MAX_BYTES,
    ADK_ROUTER_MODEL,
    ADK_TITLE_MODEL,
    ADK_WEB_TOOL_MAX_CHARS,
    GOOGLE_API_KEY,
    REPO_ROOT,
)
from .services import get_artifact_service, APP_NAME, USER_ID
from . import uploads

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


DEFAULT_BROWSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) "
        "Gecko/20100101 Firefox/132.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}
ARTIFACT_TOOL_MAX_CHARS = 8000


COUNCIL_MEMBER_INSTRUCTION = (
    "You are a voting member of the LLM Council. "
    "Provide deeply reasoned answers, cite the tools you called, and explain tradeoffs. "
    "Available tools: `google_search` (use this first to find sources), `browse_web` for "
    "fetching a URL AFTER you have a specific target, `read_repository_file` for project files, "
    "and `read_uploaded_artifact` for user-provided attachments. Prefer google_search as your "
    "first step; only call browse_web with a specific URL. Avoid guessing links or issuing "
    "repeated fetches to 404/403 pages."
)

STAGE1_PROMPT_TEMPLATE = """{history_section}You are {agent_name}, a council member asked to address:

Latest question: {user_query}

Deliver a thoughtful, tool-supported response. Explicitly mention if you used browsing or file
reading tools and summarize the evidence you relied on. Start with `google_search` to find sources; only call `browse_web` when you have a specific URL to inspect. Do not guess or spam URLs."""

STAGE2_PROMPT_TEMPLATE = """{history_section}You are {agent_name}, reviewing anonymized council responses to:

Latest question: {user_query}

{responses_text}

Requirements:
1. Give a short critique for each response (strengths + weaknesses).
2. Finish with a FINAL RANKING list that matches the exact format shown below.

FINAL RANKING:
1. Response X
2. Response Y
3. Response Z
"""

STAGE3_PROMPT_TEMPLATE = """{history_section}You are the Chairman who must synthesize the council's work.

Latest Question:
{user_query}

STAGE 1 RESPONSES:
{stage1_text}

STAGE 2 PEER REVIEWS:
{stage2_text}

Produce a single final answer that reflects consensus, resolves disagreements, and references the
strongest arguments that emerged."""

TITLE_PROMPT_TEMPLATE = """Generate a concise 3-5 word title (no quotes) that summarizes this question:

{user_query}
"""

ROUTER_PROMPT_TEMPLATE = """{history_section}User question: {user_query}

Decide whether to run the full council (multi-stage deliberation) or just have a single analyst respond directly.

Return JSON exactly in the form:
{{
  "mode": "single" | "council",
  "reason": "<brief rationale>"
}}

Use "single" for trivial greetings, acknowledgements, or clearly scoped facts.
Use "council" when the user needs multi-step reasoning, comparisons, tool usage, or oversight.
"""

def _format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            content = message.get("content", "").strip()
            if content:
                lines.append(f"User: {content}")
            attachments = message.get("attachments") or []
            if attachments:
                lines.append(_format_attachments_summary(attachments))
        elif role == "assistant":
            stage3 = message.get("stage3") or {}
            response = stage3.get("response") or stage3.get("content") or ""
            if response:
                lines.append(f"Assistant: {response.strip()}")
    return "\n".join(lines)


def _build_history_section(messages: List[Dict[str, Any]]) -> str:
    text = _format_conversation_history(messages)
    if not text:
        return "This is the first exchange between the user and the council.\n\n"
    return f"Conversation so far:\n{text}\n\n"


def _format_attachments_summary(attachments: List[Dict[str, Any]]) -> str:
    lines = ["Attachments provided:"]
    for attachment in attachments:
        filename = attachment.get("filename") or "file"
        mime_type = attachment.get("mime_type") or "unknown"
        size = attachment.get("size")
        path = attachment.get("relative_path")
        line = f"- {filename} ({mime_type}"
        if size is not None:
            line += f", {size} bytes"
        line += ")"
        if path:
            line += f" [path: {path}]"
        lines.append(line)
        excerpt = attachment.get("text_excerpt")
        if excerpt:
            excerpt_clean = excerpt.strip()
            lines.append(f"  Excerpt: {excerpt_clean}")
    return "\n".join(lines)


def browse_web(
    url: str,
    query: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Simple HTTP GET tool for agents."""
    limit = max_chars or ADK_WEB_TOOL_MAX_CHARS
    headers = _build_browse_headers(url)
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network/tool failures
        logger.warning("browse_web failed for %s: %s", url, exc)
        return {"url": url, "error": str(exc)}

    cleaned = _clean_html(response.text)
    snippet = _focus_on_query(cleaned, query, limit)

    return {
        "url": str(response.url),
        "status": response.status_code,
        "content": snippet,
        "truncated": len(cleaned) > len(snippet),
    }


def read_repository_file(
    relative_path: str,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Expose repository files to the council agents."""
    limit = max_chars or ADK_FILE_TOOL_MAX_BYTES
    base = ADK_ALLOWED_FILE_ROOT
    target = (base / relative_path).resolve()

    if base != target and base not in target.parents:
        raise ValueError("Access outside the allowed project root is not permitted.")

    if not target.exists():
        return {"path": relative_path, "error": "File not found"}

    if target.is_dir():
        children = sorted(p.name for p in target.iterdir())
        display_path = _display_path(target)
        return {"path": display_path, "directory": True, "children": children[:50]}

    text = target.read_text(errors="ignore")
    truncated = len(text) > limit
    return {
        "path": _display_path(target),
        "content": text[:limit],
        "truncated": truncated,
    }


async def read_uploaded_artifact(
    attachment_id: str,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    record = uploads.load_attachment_record(attachment_id)
    if not record:
        return {"error": "Attachment not found."}

    artifact_service = get_artifact_service()
    part = await artifact_service.load_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        filename=record["artifact_key"],
        session_id=None,
    )
    if part is None:
        return {"error": "Attachment content is no longer available."}

    text = _part_to_text(part, record.get("filename"), record.get("mime_type"))
    if text is None:
        return {
            "attachment_id": attachment_id,
            "filename": record["filename"],
            "mime_type": record["mime_type"],
            "error": "Attachment is binary or unsupported for inline reading.",
            "canonical_uri": record.get("canonical_uri"),
        }

    limit = max_chars or ARTIFACT_TOOL_MAX_CHARS
    snippet = text[:limit]
    return {
        "attachment_id": attachment_id,
        "filename": record["filename"],
        "mime_type": record["mime_type"],
        "text": snippet,
        "truncated": len(text) > len(snippet),
        "canonical_uri": record.get("canonical_uri"),
    }


BROWSE_WEB_TOOL = FunctionTool(browse_web)
READ_FILE_TOOL = FunctionTool(read_repository_file)
READ_ARTIFACT_TOOL = FunctionTool(read_uploaded_artifact)


class AdkCouncilRuntime:
    """Encapsulates the ADK-based council orchestration."""

    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY must be set to run the ADK council.")

        self._base_tools = [BROWSE_WEB_TOOL, READ_FILE_TOOL, READ_ARTIFACT_TOOL]
        self._google_search_tool = GoogleSearchTool(
            bypass_multi_tools_limit=True
        )
        self._app_name = APP_NAME
        self._user_id = USER_ID
        self._session_service = InMemorySessionService()
        self._memory_service = InMemoryMemoryService()
        self._artifact_service = get_artifact_service()
        self._council_agents = [
            self._build_member_agent(member, index)
            for index, member in enumerate(ADK_COUNCIL_MEMBERS, start=1)
        ]
        self._chairman = Agent(
            model=_resolve_model_spec(ADK_CHAIRMAN_MODEL),
            name="Chairman",
            description="Synthesizes the council output into a final answer.",
            instruction="Combine peer work into a cohesive, well-balanced response.",
            tools=self._tools_for_model(ADK_CHAIRMAN_MODEL),
        )
        self._title_agent = Agent(
            model=_resolve_model_spec(ADK_TITLE_MODEL),
            name="TitleScribe",
            description="Creates concise titles for council sessions.",
            instruction="Respond with 3-5 word descriptive titles. No punctuation.",
            tools=[],
        )
        self._router_agent = Agent(
            model=_resolve_model_spec(ADK_ROUTER_MODEL),
            name="Delegator",
            description="Decides whether to run single-agent or council workflow.",
            instruction="Output JSON specifying routing decision.",
            tools=[],
        )

    async def collect_stage1(
        self,
        user_query: str,
        history_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        timer = time.perf_counter()
        history_section = _build_history_section(history_messages)
        prompts = [
            _format_prompt(
                STAGE1_PROMPT_TEMPLATE,
                agent_name=agent.name,
                user_query=user_query,
                history_section=history_section,
            )
            for agent in self._council_agents
        ]
        results = await self._gather_agent_runs(prompts)
        stage1: List[Dict[str, Any]] = []
        for agent, text in zip(self._council_agents, results):
            if text:
                stage1.append({"model": agent.name, "response": text})
        logger.info(
            "stage1 collected responses=%d duration=%.2fs",
            len(stage1),
            time.perf_counter() - timer,
        )
        return stage1

    async def collect_stage2(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        timer = time.perf_counter()
        labels = [f"Response {chr(65 + i)}" for i in range(len(stage1_results))]
        label_to_model = {
            label: result["model"]
            for label, result in zip(labels, stage1_results)
        }
        responses_text = "\n\n".join(
            f"{label}:\n{result['response']}"
            for label, result in zip(labels, stage1_results)
        )
        history_section = _build_history_section(history_messages)
        prompts = [
            _format_prompt(
                STAGE2_PROMPT_TEMPLATE,
                agent_name=agent.name,
                user_query=user_query,
                responses_text=responses_text,
                history_section=history_section,
            )
            for agent in self._council_agents
        ]
        raw_rankings = await self._gather_agent_runs(prompts)
        stage2: List[Dict[str, Any]] = []
        for agent, ranking_text in zip(self._council_agents, raw_rankings):
            if not ranking_text:
                continue
            parsed = parse_ranking_from_text(ranking_text)
            stage2.append({
                "model": agent.name,
                "ranking": ranking_text,
                "parsed_ranking": parsed,
            })
        logger.info(
            "stage2 collected rankings=%d duration=%.2fs",
            len(stage2),
            time.perf_counter() - timer,
        )
        return stage2, label_to_model

    async def synthesize_final(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        timer = time.perf_counter()
        stage1_text = "\n\n".join(
            f"{item['model']}:\n{item['response']}" for item in stage1_results
        )
        stage2_text = "\n\n".join(
            f"{item['model']}:\n{item['ranking']}" for item in stage2_results
        )
        history_section = _build_history_section(history_messages)
        prompt = _format_prompt(
            STAGE3_PROMPT_TEMPLATE,
            user_query=user_query,
            stage1_text=stage1_text or "No responses collected.",
            stage2_text=stage2_text or "No peer reviews collected.",
            history_section=history_section,
        )
        text = await self._run_agent(self._chairman, prompt)
        logger.info(
            "stage3 synthesis complete duration=%.2fs",
            time.perf_counter() - timer,
        )
        return {
            "model": self._chairman.name,
            "response": text or "Unable to synthesize a response at this time.",
        }

    async def generate_title(self, user_query: str) -> str:
        prompt = _format_prompt(TITLE_PROMPT_TEMPLATE, user_query=user_query)
        text = await self._run_agent(self._title_agent, prompt)
        if not text:
            return "New Conversation"
        clean = text.strip().strip("\"'")
        return clean[:60] or "New Conversation"

    async def quick_response(
        self,
        user_query: str,
        history_messages: List[Dict[str, Any]],
        allow_tools: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self._council_agents:
            raise RuntimeError("No council agents configured.")
        timer = time.perf_counter()
        agent = self._council_agents[0]
        agent_for_run = agent.clone()
        if not getattr(agent_for_run, "tools", None):
            agent_for_run.tools = list(agent.tools or [])
        history_section = _build_history_section(history_messages)
        prompt = _format_prompt(
            STAGE1_PROMPT_TEMPLATE,
            agent_name=agent.name,
            user_query=user_query,
            history_section=history_section,
        )
        if not allow_tools:
            prompt = f"{prompt}\n\nRestriction: This is a simple or low-complexity message. Do not call any tools or functions. Respond directly and concisely without browsing or searching."
        response_text = await self._run_agent(agent_for_run, prompt)
        text = response_text or "Unable to provide a response at this time."
        stage_entry = {"model": agent.name, "response": text}
        logger.info(
            "single agent quick response model=%s allow_tools=%s duration=%.2fs",
            agent.name,
            allow_tools,
            time.perf_counter() - timer,
        )
        return stage_entry, {
            "model": agent.name,
            "response": text,
        }

    async def route_request(
        self,
        user_query: str,
        history_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        history_section = _build_history_section(history_messages)
        prompt = _format_prompt(
            ROUTER_PROMPT_TEMPLATE,
            user_query=user_query,
            history_section=history_section,
        )
        timer = time.perf_counter()
        decision_text = await self._run_agent(self._router_agent, prompt)
        decision = _parse_route_decision(decision_text)
        logger.info(
            "router decision mode=%s reason=%s duration=%.2fs",
            decision.get("mode"),
            decision.get("reason"),
            time.perf_counter() - timer,
        )
        return decision

    async def _gather_agent_runs(self, prompts: List[str]) -> List[Optional[str]]:
        tasks = [
            self._run_agent(agent, prompt)
            for agent, prompt in zip(self._council_agents, prompts)
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[Optional[str]] = []
        for agent, output in zip(self._council_agents, gathered):
            if isinstance(output, Exception):
                logger.exception("Agent %s failed: %s", agent.name, output)
                results.append(None)
                continue
            results.append(output)
        return results

    async def _run_agent(self, agent: Agent, prompt: str) -> Optional[str]:
        timer = time.perf_counter()
        session = await self._session_service.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=str(uuid.uuid4()),
        )
        runner = Runner(
            app_name=self._app_name,
            agent=agent.clone(),
            session_service=self._session_service,
            memory_service=self._memory_service,
            artifact_service=self._artifact_service,
        )
        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )
        final_text: Optional[str] = None
        try:
            async for event in runner.run_async(
                user_id=self._user_id,
                session_id=session.id,
                new_message=message,
            ):
                if event.author == "user":
                    continue
                text = _content_to_text(event.content)
                if text:
                    final_text = text
        except Exception as exc:  # pragma: no cover - upstream failures
            logger.exception("Failed to invoke %s: %s", agent.name, exc)
            return None
        finally:
            await self._session_service.delete_session(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session.id,
            )
        logger.info(
            "agent run complete agent=%s duration=%.2fs has_response=%s",
            agent.name,
            time.perf_counter() - timer,
            bool(final_text),
        )
        return final_text

    def _build_member_agent(self, member: Dict[str, str], index: int) -> Agent:
        name = member.get("name") or f"Agent {index}"
        model = member.get("model")
        if not model:
            raise ValueError("Each ADK council member must define a model.")
        return Agent(
            model=_resolve_model_spec(model),
            name=name,
            description=f"{name} uses {model} to contribute unique insights.",
            instruction=COUNCIL_MEMBER_INSTRUCTION,
            tools=self._tools_for_model(model),
        )

    def _tools_for_model(self, model: str) -> List[Any]:
        tools: List[Any] = list(self._base_tools)
        if is_gemini_model(model):
            tools.append(self._google_search_tool)
        return tools

def _resolve_model_spec(model: Union[str, LiteLlm, None]) -> Union[str, LiteLlm, None]:
    if not model or isinstance(model, LiteLlm):
        return model
    if model.startswith("openrouter/"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set to use OpenRouter-backed agents."
            )
        api_base = os.getenv("OPENROUTER_API_BASE", OPENROUTER_DEFAULT_API_BASE)
        return LiteLlm(model=model, api_key=api_key, api_base=api_base)
    return model


def _parse_route_decision(raw_text: Optional[str]) -> Dict[str, Any]:
    default = {"mode": "council", "reason": "router_default"}
    if not raw_text:
        return default
    try:
        data = json.loads(raw_text)
        mode = str(data.get("mode", "council")).strip().lower()
        reason = data.get("reason") or "router_response"
        if mode not in ("single", "council"):
            mode = "council"
        return {"mode": mode, "reason": reason}
    except Exception:
        cleaned = raw_text.strip().lower()
        if "single" in cleaned and "council" not in cleaned:
            return {"mode": "single", "reason": raw_text[:200]}
        return default


def build_prompt_with_attachments(content: str, attachments: Optional[List[Dict[str, Any]]]) -> str:
    if not attachments:
        return content

    lines = [content.strip(), "", "Attachments:"]
    for attachment in attachments:
        filename = attachment.get("filename") or "file"
        mime_type = attachment.get("mime_type") or "unknown"
        artifact_id = attachment.get("id")
        canonical_uri = attachment.get("canonical_uri")
        line = f"- {filename} ({mime_type})"
        if artifact_id:
            line += f" [Attachment ID: {artifact_id}]"
        if canonical_uri:
            line += f" [Artifact URI: {canonical_uri}]"
        lines.append(line)
        excerpt = attachment.get("text_excerpt")
        if excerpt:
            excerpt_clean = excerpt.strip()
            if len(excerpt_clean) > 1000:
                excerpt_clean = excerpt_clean[:1000] + "..."
            lines.append(f"  Preview: {excerpt_clean}")
        if artifact_id:
            lines.append(f"  Use read_uploaded_artifact('{artifact_id}') for full text.")
    lines.append("")
    lines.append("Use the provided Attachment IDs with read_uploaded_artifact when you need full file contents.")
    return "\n".join(lines)


def _part_to_text(part: genai_types.Part, filename: Optional[str], mime: Optional[str]) -> Optional[str]:
    if part.text:
        return part.text
    if part.inline_data and part.inline_data.data:
        return _bytes_to_text(part.inline_data.data, filename, mime)
    return None


def _bytes_to_text(data: bytes, filename: Optional[str], mime_type: Optional[str]) -> Optional[str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json"} or (mime_type or "").startswith("text/"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")
    if suffix == ".pdf" or mime_type == "application/pdf":
        if PdfReader is None:
            return None
        try:
            from io import BytesIO

            reader = PdfReader(BytesIO(data))
            chunks: List[str] = []
            for page in reader.pages:
                content = page.extract_text() or ""
                if content:
                    chunks.append(content)
            return "\n".join(chunks).strip() or None
        except Exception:
            return None
    return None


def _clean_html(raw: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]*>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _focus_on_query(text: str, query: Optional[str], limit: int) -> str:
    if not query:
        return text[:limit]

    lowered = text.lower()
    q = query.lower()
    idx = lowered.find(q)
    if idx == -1:
        return text[:limit]

    start = max(idx - limit // 4, 0)
    end = min(start + limit, len(text))
    return text[start:end]


def _display_path(target: Path) -> str:
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)


def _build_browse_headers(url: str) -> Dict[str, str]:
    headers = dict(DEFAULT_BROWSE_HEADERS)
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return headers

    if domain.endswith("wikipedia.org"):
        headers["Referer"] = "https://www.wikipedia.org/"
    elif domain.endswith("github.com"):
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        headers["Referer"] = "https://github.com/"
    return headers


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _format_prompt(template: str, **kwargs: Any) -> str:
    safe_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            safe_kwargs[key] = _escape_braces(value)
        else:
            safe_kwargs[key] = value
    return template.format(**safe_kwargs)


def _coerce_response_text(result: Any) -> Optional[str]:
    if result is None:
        return None

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in ("text", "response", "content", "output"):
            if key in result and isinstance(result[key], str):
                return result[key].strip()
        return str(result)

    if hasattr(result, "text") and isinstance(result.text, str):
        return result.text.strip()

    if isinstance(result, genai_types.Content):
        parts = []
        for part in result.parts or []:
            if getattr(part, "text", None):
                parts.append(part.text)
        if parts:
            return "\n".join(parts).strip()

    response_type = getattr(genai_types, "GenerateContentResponse", tuple())
    if response_type and isinstance(result, response_type):
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()

    return str(result).strip()


def _content_to_text(content: Optional[genai_types.Content]) -> Optional[str]:
    if content is None or not getattr(content, "parts", None):
        return None
    texts: List[str] = []
    for part in content.parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text.strip())
    combined = "\n".join(filter(None, texts)).strip()
    return combined or None


_runtime: Optional[AdkCouncilRuntime] = None


def _get_runtime() -> AdkCouncilRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AdkCouncilRuntime()
        member_desc = ", ".join(
            f"{m.get('name')}[{m.get('model')}]" for m in ADK_COUNCIL_MEMBERS
        )
        logger.info(
            "adk runtime initialized router=%s chairman=%s title=%s members=%s",
            ADK_ROUTER_MODEL,
            ADK_CHAIRMAN_MODEL,
            ADK_TITLE_MODEL,
            member_desc,
        )
    return _runtime


async def stage1_collect_responses(
    user_query: str,
    history_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return await _get_runtime().collect_stage1(user_query, history_messages)


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    history_messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    return await _get_runtime().collect_stage2(user_query, stage1_results, history_messages)


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    history_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return await _get_runtime().synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        history_messages,
    )


async def generate_conversation_title(user_query: str) -> str:
    return await _get_runtime().generate_title(user_query)


async def route_message(
    user_query: str,
    history_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    history = history_messages or []
    return await _get_runtime().route_request(user_query, history)


async def run_quick_response(
    user_query: str,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    allow_tools: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    history = history_messages or []
    return await _get_runtime().quick_response(user_query, history, allow_tools=allow_tools)


async def run_full_council(
    user_query: str,
    history_messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List, List, Dict, Dict]:
    history = history_messages or []
    stage1_results = await stage1_collect_responses(user_query, history)
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All ADK agents failed to respond. Please try again."
        }, {}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results, history)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        history,
    )
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
    }
    return stage1_results, stage2_results, stage3_result, metadata


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    model_positions: Dict[str, List[int]] = defaultdict(list)

    for ranking in stage2_results:
        parsed_ranking = ranking.get("parsed_ranking") or parse_ranking_from_text(ranking.get("ranking", ""))
        for position, label in enumerate(parsed_ranking, start=1):
            model_name = label_to_model.get(label)
            if model_name:
                model_positions[model_name].append(position)

    aggregate: List[Dict[str, Any]] = []
    for model, positions in model_positions.items():
        if not positions:
            continue
        avg_rank = sum(positions) / len(positions)
        aggregate.append({
            "model": model,
            "average_rank": round(avg_rank, 2),
            "rankings_count": len(positions),
        })

    aggregate.sort(key=lambda item: item["average_rank"])
    return aggregate


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    import re as _re
    if "FINAL RANKING" not in ranking_text.upper():
        return []
    section = ranking_text.split("FINAL RANKING", 1)[1]
    matches = _re.findall(r"\d+\.\s*(Response [A-Z])", section)
    return matches
