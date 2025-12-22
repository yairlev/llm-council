"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio
import logging
import os
import time

from . import storage, uploads
from .council import (
    run_pipeline,
    generate_conversation_title,
    build_prompt_with_attachments,
)

LOG_LEVEL = os.getenv("LLM_COUNCIL_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
else:
    root_logger.setLevel(LOG_LEVEL)

logger = logging.getLogger("llm_council.api")

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    attachments: List[str] = []


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/attachments")
async def upload_attachment(conversation_id: str, file: UploadFile = File(...)):
    """Upload a file to use as context within a conversation."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    metadata = await uploads.save_upload(conversation_id, file)
    return metadata


@app.delete("/api/conversations/{conversation_id}/attachments/{attachment_id}")
async def delete_attachment(conversation_id: str, attachment_id: str):
    """Delete a previously uploaded file that hasn't been linked to a message."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    success = await uploads.delete_attachment(conversation_id, attachment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Attachment cannot be deleted (it may be linked or missing).")
    return {"status": "deleted", "id": attachment_id}


def _get_history_messages(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract history messages (excluding the last user message if present)."""
    messages = conversation.get("messages", [])
    if messages and messages[-1].get("role") == "user":
        return messages[:-1]
    return messages


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    start_overall = time.perf_counter()

    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    attachment_ids = request.attachments or []
    attachment_records = uploads.get_attachments(conversation_id, attachment_ids)
    if len(attachment_records) != len(attachment_ids):
        raise HTTPException(status_code=400, detail="One or more attachments were not found.")

    user_prompt = build_prompt_with_attachments(request.content, attachment_records)
    logger.info(
        "message request conversation_id=%s first_message=%s attachments=%d",
        conversation_id,
        is_first_message,
        len(attachment_ids),
    )

    # Add user message and get history
    storage.add_user_message(conversation_id, request.content, attachments=attachment_records)
    uploads.mark_attachments_linked(conversation_id, attachment_ids)
    conversation = storage.get_conversation(conversation_id)
    history_messages = _get_history_messages(conversation) if conversation else []

    # Run the unified pipeline (no streaming callback for sync endpoint)
    stage1_results, stage2_results, stage3_result, metadata = await run_pipeline(
        user_prompt,
        history_messages,
    )

    logger.info(
        "pipeline complete conversation_id=%s mode=%s duration=%.2fs",
        conversation_id,
        metadata.get("mode"),
        time.perf_counter() - start_overall,
    )

    # Save to storage
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata=metadata,
    )

    # Generate title for first non-trivial message
    route_info = metadata.get("route", {})
    title_reason = route_info.get("reason", "").lower()
    if is_first_message and "trivial" not in title_reason:
        try:
            title = await generate_conversation_title(request.content)
            storage.update_conversation_title(conversation_id, title)
        except Exception as exc:
            logger.warning("title generation failed conversation_id=%s error=%s", conversation_id, exc)

    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata,
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    start_overall = time.perf_counter()

    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    attachment_ids = request.attachments or []
    attachment_records = uploads.get_attachments(conversation_id, attachment_ids)
    if len(attachment_records) != len(attachment_ids):
        raise HTTPException(status_code=400, detail="One or more attachments were not found.")

    user_prompt = build_prompt_with_attachments(request.content, attachment_records)

    async def event_generator():
        # Queue for collecting events from the pipeline callback
        event_queue: asyncio.Queue = asyncio.Queue()
        metadata_holder: Dict[str, Any] = {}

        async def on_pipeline_event(event_type: str, data: Any) -> None:
            """Callback invoked by run_pipeline for each stage transition."""
            if event_type == "route":
                logger.info(
                    "router decision (stream) conversation_id=%s mode=%s reason=%s",
                    conversation_id,
                    data.get("mode"),
                    data.get("reason"),
                )
            elif event_type == "stage1_start":
                await event_queue.put({"type": "stage1_start"})
            elif event_type == "stage1_agent_complete":
                await event_queue.put({"type": "stage1_agent_complete", "data": data})
            elif event_type == "stage1_complete":
                await event_queue.put({"type": "stage1_complete", "data": data})
            elif event_type == "stage2_start":
                await event_queue.put({"type": "stage2_start"})
            elif event_type == "stage2_agent_complete":
                await event_queue.put({"type": "stage2_agent_complete", "data": data})
            elif event_type == "stage2_complete":
                metadata_holder.update(data.get("metadata", {}))
                await event_queue.put({
                    "type": "stage2_complete",
                    "data": data.get("data", []),
                    "metadata": data.get("metadata", {}),
                })
            elif event_type == "stage3_start":
                await event_queue.put({"type": "stage3_start"})
            elif event_type == "stage3_complete":
                await event_queue.put({"type": "stage3_complete", "data": data})

        try:
            logger.info(
                "streaming message request conversation_id=%s first_message=%s attachments=%d",
                conversation_id,
                is_first_message,
                len(attachment_ids),
            )

            # Add user message and get history
            storage.add_user_message(conversation_id, request.content, attachments=attachment_records)
            uploads.mark_attachments_linked(conversation_id, attachment_ids)
            convo = storage.get_conversation(conversation_id)
            history_messages = _get_history_messages(convo) if convo else []

            # Run pipeline in a task so we can yield events as they arrive
            async def run_and_signal_done():
                result = await run_pipeline(user_prompt, history_messages, on_event=on_pipeline_event)
                await event_queue.put({"type": "_done", "result": result})
                return result

            pipeline_task = asyncio.create_task(run_and_signal_done())

            # Start title generation early (will be awaited later)
            title_task = None

            # Yield events as they arrive from the pipeline
            stage1_results = []
            stage2_results = []
            stage3_result = {}

            while True:
                event = await event_queue.get()

                if event["type"] == "_done":
                    stage1_results, stage2_results, stage3_result, metadata = event["result"]
                    break

                # Start title generation after we get routing info (via stage2_complete metadata)
                if event["type"] == "stage2_complete" and title_task is None:
                    route_info = event.get("metadata", {}).get("route", {})
                    title_reason = route_info.get("reason", "").lower()
                    if is_first_message and "trivial" not in title_reason:
                        title_task = asyncio.create_task(generate_conversation_title(request.content))

                yield f"data: {json.dumps(event)}\n\n"

            # Await pipeline completion (should already be done)
            await pipeline_task
            metadata = metadata_holder

            # Wait for title generation if started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save to storage
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata=metadata,
            )

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            logger.info(
                "streaming response stored conversation_id=%s mode=%s total_duration=%.2fs",
                conversation_id,
                metadata.get("mode"),
                time.perf_counter() - start_overall,
            )

        except Exception as e:
            logger.exception("streaming pipeline failed conversation_id=%s", conversation_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
