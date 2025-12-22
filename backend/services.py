"""Shared ADK services (artifact service, app/user identifiers)."""

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

APP_NAME = "llm-council-adk"
USER_ID = "local-user"

_artifact_service = InMemoryArtifactService()


def get_artifact_service() -> InMemoryArtifactService:
    return _artifact_service
