"""Configuration for the LLM Council."""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Repository root path (used by file-reading tools)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Google / ADK credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _normalize_model_name(model: str) -> str:
    """Map Vertex-style identifiers to provider/model strings"""
    if not model:
        return model
    model = model.strip()
    parts = model.split("/")
    if len(parts) >= 4 and parts[0] == "publishers" and parts[2] == "models":
        provider = parts[1]
        model_name = parts[3]
        return f"{provider}/{model_name}"
    return model


def _parse_adk_members(raw_value: str | None) -> List[Dict[str, str]]:
    """Parse ADK council member definitions from a comma-separated env var."""
    default = [
        {"name": "Orion", "model": "gemini-3-pro-preview"},
        {"name": "Lyra", "model": "gemini-3-flash-preview"},
        {"name": "Vega", "model": "openrouter/deepseek/deepseek-r1"},
    ]

    if not raw_value:
        return default

    members: List[Dict[str, str]] = []
    for index, chunk in enumerate(raw_value.split(","), start=1):
        chunk = chunk.strip()
        if not chunk:
            continue

        if ":" in chunk:
            label, model = chunk.split(":", 1)
        else:
            label, model = f"Agent {index}", chunk

        members.append({
            "name": label.strip(),
            "model": _normalize_model_name(model),
        })

    return members or default


# ADK-specific configuration
ADK_COUNCIL_MEMBERS = _parse_adk_members(os.getenv("ADK_COUNCIL_MEMBERS"))
ADK_CHAIRMAN_MODEL = _normalize_model_name(os.getenv("ADK_CHAIRMAN_MODEL", "gemini-3-pro-preview"))
ADK_TITLE_MODEL = _normalize_model_name(os.getenv("ADK_TITLE_MODEL", ADK_CHAIRMAN_MODEL))
ADK_FILE_TOOL_MAX_BYTES = int(os.getenv("ADK_FILE_TOOL_MAX_BYTES", "20000"))
ADK_WEB_TOOL_MAX_CHARS = int(os.getenv("ADK_WEB_TOOL_MAX_CHARS", "4000"))
ADK_ALLOWED_FILE_ROOT = Path(os.getenv("ADK_ALLOWED_FILE_ROOT", str(REPO_ROOT))).resolve()
