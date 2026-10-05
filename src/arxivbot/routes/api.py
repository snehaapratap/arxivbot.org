"""API routes for chat functionality."""

import json
import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from arxivbot.limiter import limiter
from arxivbot.services import chat_service, paper_service
from arxivbot.utils.arxiv import parse_arxiv_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


MAX_MESSAGE_LENGTH = 4000


class ChatRequestBody(BaseModel):
    """Request body for chat endpoint (aliased to avoid conflict with FastAPI Request)."""

    paper_id: str
    chat_slug: str | None = None
    message: str
    # How the answer is pitched: everyday words (default) or the field's own terms.
    # "legacy" is the pre-nf4 prompt, used by the /old snapshot UI.
    level: Literal["plain", "technical", "legacy"] = "plain"
    # Continue chat_slug as a new copy instead of appending to it (someone else's shared chat)
    fork: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message must not be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message must be at most {MAX_MESSAGE_LENGTH} characters")
        return v


def _user_error_message(e: Exception) -> str:
    """Error text safe to show users; hides raw LLM/provider exceptions."""
    if isinstance(e, paper_service.PaperFetchError):
        return str(e)
    return "Something went wrong while ArxivBot was answering. Try asking again."


class Citation(BaseModel):
    """A citation from the paper."""

    text: str
    page: int | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    chat_slug: str
    response: str
    citations: list[Citation] = []


class StatusResponse(BaseModel):
    """Response from status endpoint."""

    status: str
    progress: int
    error: str | None = None


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat(request: Request, body: ChatRequestBody):
    """
    Send a message and get a response.

    If chat_slug is None, creates a new chat.
    Otherwise, continues an existing chat.
    Nothing is persisted unless the answer succeeds.
    """
    # Validate paper ID
    parsed = parse_arxiv_id(body.paper_id)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid arXiv ID: {body.paper_id}")
    paper_id = parsed.paper_id

    # Get or create chat
    chat = None
    if body.chat_slug:
        chat = await chat_service.get_chat_by_slug(body.chat_slug)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        stored = parse_arxiv_id(chat.paper_id)
        if not stored or stored.paper_id != paper_id:
            raise HTTPException(status_code=400, detail="Paper ID mismatch")

    # Get chat history for context
    chat_history = []
    if chat:
        messages = await chat_service.get_messages(chat.id)
        chat_history = [{"role": m.role, "content": m.content} for m in messages]

    try:
        # Query the paper
        result = await paper_service.query_paper(
            paper_id=paper_id,
            question=body.message,
            chat_history=chat_history,
            level=body.level,
        )

        # Persist the turn (and the chat, if new) only once the answer succeeded
        chat_slug = await chat_service.save_turn(
            body.message,
            result["answer"],
            chat=chat,
            paper_id=paper_id,
            fork_from=chat if body.fork else None,
        )

        return ChatResponse(
            chat_slug=chat_slug,
            response=result["answer"],
            citations=[
                Citation(text=c["text"], page=c.get("page"))
                for c in result.get("citations", [])
            ],
        )

    except Exception as e:
        logger.exception(f"Error querying paper: {e}")
        raise HTTPException(status_code=500, detail=_user_error_message(e))


@router.post("/chat/stream")
@limiter.limit("5/minute")
async def chat_stream(request: Request, body: ChatRequestBody):
    """
    Send a message and get a streaming response via SSE.

    If chat_slug is None, creates a new chat.
    Otherwise, continues an existing chat.
    Nothing is persisted unless the answer succeeds.
    """
    # Validate paper ID
    parsed = parse_arxiv_id(body.paper_id)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid arXiv ID: {body.paper_id}")
    paper_id = parsed.paper_id

    # Get or create chat
    chat = None
    if body.chat_slug:
        chat = await chat_service.get_chat_by_slug(body.chat_slug)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        stored = parse_arxiv_id(chat.paper_id)
        if not stored or stored.paper_id != paper_id:
            raise HTTPException(status_code=400, detail="Paper ID mismatch")

    # Get chat history for context
    chat_history = []
    if chat:
        messages = await chat_service.get_messages(chat.id)
        chat_history = [{"role": m.role, "content": m.content} for m in messages]

    async def generate():
        full_response = []
        try:
            # Stream the response
            async for chunk in paper_service.query_paper_stream(
                paper_id=paper_id,
                question=body.message,
                chat_history=chat_history,
                level=body.level,
            ):
                full_response.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            # Persist the turn (and the chat, if new) only after a complete answer
            complete_response = "".join(full_response)
            chat_slug = await chat_service.save_turn(
                body.message,
                complete_response,
                chat=chat,
                paper_id=paper_id,
                fork_from=chat if body.fork else None,
            )

            # Send chat slug only once it exists in the database
            yield f"data: {json.dumps({'type': 'meta', 'chat_slug': chat_slug})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception(f"Error in streaming response: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': _user_error_message(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status/{paper_id:path}", response_model=StatusResponse)
async def get_status(paper_id: str):
    """
    Get the indexing status for a paper.

    Read-only: never starts indexing. POST /api/index/<id> is the only trigger.
    """
    # Validate paper ID
    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid arXiv ID: {paper_id}")
    paper_id = parsed.paper_id

    status = await paper_service.get_indexing_status(paper_id)

    return StatusResponse(
        status=status.get("status", "unknown"),
        progress=status.get("progress", 0),
        error=status.get("error"),
    )


@router.post("/index/{paper_id:path}")
@limiter.limit("10/minute")
async def start_indexing(
    request: Request,
    paper_id: str,
    background_tasks: BackgroundTasks,
    auto: bool = False,
):
    """
    Start indexing a paper in the background.

    With auto (page load), a failed paper is only retried once its error
    cooldown has passed. Without it (manual Retry), any error is retried.
    """
    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid arXiv ID: {paper_id}")
    paper_id = parsed.paper_id

    status = await paper_service.get_indexing_status(paper_id)

    if status["status"] == "complete":
        return {"message": "Already indexed"}

    if status["status"] not in ("not_started", "error"):
        return {"message": "Indexing in progress"}

    if auto and status["status"] == "error" and not paper_service.should_retry_after_error(status):
        return {"message": "Retry cooldown active"}

    paper_service.mark_indexing_started(paper_id)
    background_tasks.add_task(paper_service.index_paper, paper_id)
    return {"message": "Indexing started"}
