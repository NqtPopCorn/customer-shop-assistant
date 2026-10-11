from typing import List, Literal, Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from customer_shop_assistant.core.config import settings
from customer_shop_assistant.services.chat_service import chat_service

router = APIRouter(prefix="/api/v1", tags=["Assistant"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Message from the customer", min_length=1)
    history: Optional[List[ChatMessage]] = Field(
        default=None, description="Previous messages in the conversation"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional custom system prompt"
    )


class ChatResponse(BaseModel):
    reply: str
    model: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        model=settings.ollama_model,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        history_dicts = (
            [msg.model_dump() for msg in request.history] if request.history else None
        )
        reply = await chat_service.generate_response(
            message=request.message,
            history=history_dicts,
            custom_system_prompt=request.system_prompt,
        )
        return ChatResponse(reply=reply, model=settings.ollama_model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate response: {str(exc)}",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    try:
        history_dicts = (
            [msg.model_dump() for msg in request.history] if request.history else None
        )

        async def token_generator():
            try:
                async for chunk in chat_service.stream_response(
                    message=request.message,
                    history=history_dicts,
                    custom_system_prompt=request.system_prompt,
                ):
                    yield chunk
            except Exception as stream_exc:
                yield f"\n[Error: {str(stream_exc)}]"

        return StreamingResponse(token_generator(), media_type="text/plain")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initiate streaming: {str(exc)}",
        ) from exc
