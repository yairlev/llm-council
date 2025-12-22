"""Attachment management backed by ADK artifacts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile
from google.genai import types as genai_types

from .services import get_artifact_service, APP_NAME, USER_ID
from .config import REPO_ROOT

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

TEXT_SNIPPET_LIMIT = 4000
UPLOADS_ROOT = REPO_ROOT / "data" / "uploads"


def _conversation_dir(conversation_id: str) -> Path:
    path = UPLOADS_ROOT / conversation_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(conversation_id: str) -> Path:
    return _conversation_dir(conversation_id) / "manifest.json"


def _load_manifest(conversation_id: str) -> Dict[str, Dict]:
    manifest_file = _manifest_path(conversation_id)
    if not manifest_file.exists():
        return {}
    return json.loads(manifest_file.read_text())


def _save_manifest(conversation_id: str, manifest: Dict[str, Dict]) -> None:
    manifest_file = _manifest_path(conversation_id)
    manifest_file.write_text(json.dumps(manifest, indent=2))


def _sanitize_filename(filename: str) -> str:
    keep = (" ", ".", "_", "-")
    cleaned = "".join(ch for ch in filename if ch.isalnum() or ch in keep)
    return cleaned or "upload"


async def save_upload(conversation_id: str, upload: UploadFile) -> Dict[str, Any]:
    artifact_service = get_artifact_service()
    manifest = _load_manifest(conversation_id)
    simple_id = f"{conversation_id}:{uuid.uuid4()}"
    original_name = upload.filename or "upload"
    filename = _sanitize_filename(original_name)

    data = upload.file.read()
    mime_type = upload.content_type or "application/octet-stream"
    excerpt = _extract_text_excerpt(data, filename, mime_type)

    artifact_filename = f"user:{conversation_id}:{simple_id}_{filename}"
    part = genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type=mime_type,
            data=data,
        )
    )
    version = await artifact_service.save_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        filename=artifact_filename,
        artifact=part,
        session_id=None,
        custom_metadata={
            "conversation_id": conversation_id,
            "original_filename": original_name,
        },
    )
    artifact_version = await artifact_service.get_artifact_version(
        app_name=APP_NAME,
        user_id=USER_ID,
        filename=artifact_filename,
        session_id=None,
        version=version,
    )
    metadata = {
        "id": simple_id,
        "conversation_id": conversation_id,
        "filename": original_name,
        "mime_type": mime_type,
        "size": len(data),
        "artifact_key": artifact_filename,
        "canonical_uri": artifact_version.canonical_uri if artifact_version else None,
        "text_excerpt": excerpt,
        "linked": False,
        "created_at": datetime.utcnow().isoformat(),
    }
    manifest[simple_id] = metadata
    _save_manifest(conversation_id, manifest)
    return metadata


def get_attachments(conversation_id: str, attachment_ids: List[str]) -> List[Dict[str, Any]]:
    if not attachment_ids:
        return []
    manifest = _load_manifest(conversation_id)
    items: List[Dict[str, Any]] = []
    for attachment_id in attachment_ids:
        entry = manifest.get(attachment_id)
        if entry:
            items.append(entry)
    return items


def load_attachment_record(attachment_id: str) -> Optional[Dict[str, Any]]:
    conv_id, _, _ = attachment_id.partition(":")
    if not conv_id:
        return None
    manifest = _load_manifest(conv_id)
    return manifest.get(attachment_id)


def mark_attachments_linked(conversation_id: str, attachment_ids: List[str]) -> None:
    if not attachment_ids:
        return
    manifest = _load_manifest(conversation_id)
    changed = False
    for attachment_id in attachment_ids:
        entry = manifest.get(attachment_id)
        if entry and not entry.get("linked"):
            entry["linked"] = True
            changed = True
    if changed:
        _save_manifest(conversation_id, manifest)


async def delete_attachment(conversation_id: str, attachment_id: str) -> bool:
    manifest = _load_manifest(conversation_id)
    entry = manifest.get(attachment_id)
    if not entry or entry.get("linked"):
        return False

    artifact_service = get_artifact_service()
    await artifact_service.delete_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        filename=entry["artifact_key"],
        session_id=None,
    )
    manifest.pop(attachment_id, None)
    _save_manifest(conversation_id, manifest)
    return True


def _extract_text_excerpt(data: bytes, filename: str, mime_type: str) -> Optional[str]:
    suffix = Path(filename).suffix.lower()
    text: Optional[str] = None
    if (
        suffix in {".txt", ".md", ".py", ".js", ".ts", ".json"}
        or mime_type.startswith("text/")
    ):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except UnicodeDecodeError:
                text = None
    elif suffix == ".pdf" or mime_type == "application/pdf":
        if PdfReader is not None:
            try:
                from io import BytesIO

                reader = PdfReader(BytesIO(data))
                pages = []
                for page in reader.pages:
                    content = page.extract_text() or ""
                    if content:
                        pages.append(content)
                text = "\n".join(pages)
            except Exception:
                text = None
    if not text:
        return None
    snippet = text.strip()
    if len(snippet) > TEXT_SNIPPET_LIMIT:
        snippet = snippet[:TEXT_SNIPPET_LIMIT] + "..."
    return snippet
