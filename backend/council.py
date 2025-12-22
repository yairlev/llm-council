"""Google ADK-backed 3-stage council orchestration."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
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
    ADK_TITLE_MODEL,
    ADK_WEB_TOOL_MAX_CHARS,
    GOOGLE_API_KEY,
    REPO_ROOT,
)

logger = logging.getLogger(__name__)


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


COUNCIL_MEMBER_INSTRUCTION = (
    "You are a voting member of the LLM Council. "
    "Provide deeply reasoned answers, cite the tools you called, and explain tradeoffs. "
    "Available tools: `browse_web` for fetching URLs and `read_repository_file` for "
    "reading project files. Use them whenever you need current context or source material."
)

STAGE1_PROMPT_TEMPLATE = """{history_section}You are {agent_name}, a council member asked to address:

Latest question: {user_query}

Deliver a thoughtful, tool-supported response. Explicitly mention if you used browsing or file
reading tools and summarize the evidence you relied on."""

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

def _format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            content = message.get("content", "").strip()
            if content:
                lines.append(f"User: {content}")
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


BROWSE_WEB_TOOL = FunctionTool(browse_web)
READ_FILE_TOOL = FunctionTool(read_repository_file)


class AdkCouncilRuntime:
    """Encapsulates the ADK-based council orchestration."""

    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY must be set to run the ADK council.")

        self._base_tools = [BROWSE_WEB_TOOL, READ_FILE_TOOL]
        self._google_search_tool = google_search
        self._app_name = "llm-council-adk"
        self._user_id = "local-user"
        self._session_service = InMemorySessionService()
        self._memory_service = InMemoryMemoryService()
        self._artifact_service = InMemoryArtifactService()
        self._council_agents = [
            self._build_member_agent(member, index)
            for index, member in enumerate(ADK_COUNCIL_MEMBERS, start=1)
        ]
        self._chairman = Agent(
            model=ADK_CHAIRMAN_MODEL,
            name="Chairman",
            description="Synthesizes the council output into a final answer.",
            instruction="Combine peer work into a cohesive, well-balanced response.",
            tools=self._tools_for_model(ADK_CHAIRMAN_MODEL),
        )
        self._title_agent = Agent(
            model=ADK_TITLE_MODEL,
            name="TitleScribe",
            description="Creates concise titles for council sessions.",
            instruction="Respond with 3-5 word descriptive titles. No punctuation.",
        )

    async def collect_stage1(
        self,
        user_query: str,
        history_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        history_section = _build_history_section(history_messages)
        prompts = [
            STAGE1_PROMPT_TEMPLATE.format(
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
        return stage1

    async def collect_stage2(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
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
        prompt = STAGE2_PROMPT_TEMPLATE.format(
            agent_name="{agent_name}",
            user_query=user_query,
            responses_text=responses_text,
            history_section=history_section,
        )
        prompts = [
            prompt.format(agent_name=agent.name)
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
        return stage2, label_to_model

    async def synthesize_final(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        stage1_text = "\n\n".join(
            f"{item['model']}:\n{item['response']}" for item in stage1_results
        )
        stage2_text = "\n\n".join(
            f"{item['model']}:\n{item['ranking']}" for item in stage2_results
        )
        history_section = _build_history_section(history_messages)
        prompt = STAGE3_PROMPT_TEMPLATE.format(
            user_query=user_query,
            stage1_text=stage1_text or "No responses collected.",
            stage2_text=stage2_text or "No peer reviews collected.",
            history_section=history_section,
        )
        text = await self._run_agent(self._chairman, prompt)
        return {
            "model": self._chairman.name,
            "response": text or "Unable to synthesize a response at this time.",
        }

    async def generate_title(self, user_query: str) -> str:
        prompt = TITLE_PROMPT_TEMPLATE.format(user_query=user_query)
        text = await self._run_agent(self._title_agent, prompt)
        if not text:
            return "New Conversation"
        clean = text.strip().strip("\"'")
        return clean[:60] or "New Conversation"

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
        return final_text

    def _build_member_agent(self, member: Dict[str, str], index: int) -> Agent:
        name = member.get("name") or f"Agent {index}"
        model = member.get("model")
        if not model:
            raise ValueError("Each ADK council member must define a model.")
        return Agent(
            model=model,
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
