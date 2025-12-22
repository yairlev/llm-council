"""Utility helpers for managing user-uploaded files."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

from .config import REPO_ROOT

UPLOADS_ROOT = REPO_ROOT / "data" / "uploads"
TEXT_SNIPPET_LIMIT = 4000


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
    data = json.loads(manifest_file.read_text())
    return data


def _save_manifest(conversation_id: str, manifest: Dict[str, Dict]) -> None:
    manifest_file = _manifest_path(conversation_id)
    manifest_file.write_text(json.dumps(manifest, indent=2))


def _sanitize_filename(filename: str) -> str:
    keep = (" ", ".", "_", "-")
    cleaned = "".join(ch for ch in filename if ch.isalnum() or ch in keep)
    return cleaned or "upload"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_upload(conversation_id: str, upload: UploadFile) -> Dict[str, Any]:
    manifest = _load_manifest(conversation_id)
    file_id = str(uuid.uuid4())
    filename = _sanitize_filename(upload.filename or "upload")
    dest_path = _conversation_dir(conversation_id) / f"{file_id}_{filename}"

    upload.file.seek(0)
    with dest_path.open("wb") as dest:
        shutil.copyfileobj(upload.file, dest)

    mime_type = upload.content_type or "application/octet-stream"
    excerpt, text_path = _extract_text(dest_path, mime_type)
    metadata = {
        "id": file_id,
        "filename": filename,
        "mime_type": mime_type,
        "size": dest_path.stat().st_size,
        "relative_path": _relative_path(dest_path),
        "text_excerpt": excerpt,
        "text_path": _relative_path(text_path) if text_path else None,
        "linked": False,
        "created_at": datetime.utcnow().isoformat(),
    }
    manifest[file_id] = metadata
    _save_manifest(conversation_id, manifest)
    return metadata


def get_attachment(conversation_id: str, attachment_id: str) -> Optional[Dict[str, Any]]:
    manifest = _load_manifest(conversation_id)
    return manifest.get(attachment_id)


def get_attachments(conversation_id: str, attachment_ids: List[str]) -> List[Dict[str, Any]]:
    manifest = _load_manifest(conversation_id)
    records: List[Dict[str, Any]] = []
    for attachment_id in attachment_ids:
        entry = manifest.get(attachment_id)
        if entry:
            records.append(entry)
    return records


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


def delete_attachment(conversation_id: str, attachment_id: str) -> bool:
    manifest = _load_manifest(conversation_id)
    entry = manifest.get(attachment_id)
    if not entry or entry.get("linked"):
        return False

    path = REPO_ROOT / entry["relative_path"]
    if path.exists():
        path.unlink()
    text_path = entry.get("text_path")
    if text_path:
        txt_file = REPO_ROOT / text_path
        if txt_file.exists():
            txt_file.unlink()
    manifest.pop(attachment_id, None)
    _save_manifest(conversation_id, manifest)
    return True


def _extract_text(path: Path, mime_type: str) -> tuple[Optional[str], Optional[Path]]:
    suffix = path.suffix.lower()
    text: Optional[str] = None
    if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json"} or mime_type.startswith("text/"):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            text = None
    elif suffix == ".pdf" or mime_type == "application/pdf":
        if PdfReader is None:
            text = None
        else:
            try:
                reader = PdfReader(str(path))
                chunks = []
                for page in reader.pages:
                    content = page.extract_text() or ""
                    if content:
                        chunks.append(content)
                text = "\n".join(chunks)
            except Exception:
                text = None

    if not text:
        return None, None

    snippet = text.strip()
    if len(snippet) > TEXT_SNIPPET_LIMIT:
        snippet = snippet[:TEXT_SNIPPET_LIMIT] + "..."
    preview_path = path.with_suffix(path.suffix + ".txt")
    preview_path.write_text(text, errors="ignore")
    return snippet, preview_path
