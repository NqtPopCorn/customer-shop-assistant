from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from customer_shop_assistant.services.agent_service import ShopAssistantAgent

router = APIRouter(prefix="/api/v1", tags=["Assistant"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model: str


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    active_model = (
        settings.OLLAMA_MODEL
        if settings.LLM_PROVIDER == "ollama"
        else settings.LMSTUDIO_MODEL
    )
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        model=active_model,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    try:
        agent = ShopAssistantAgent(settings=settings)
        return await agent.chat(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate response: {exc}",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    try:
        agent = ShopAssistantAgent(settings=settings)

        async def token_generator() -> AsyncIterator[str]:
            try:
                async for chunk in agent.chat_stream(request):
                    c_type = chunk.get("type")
                    if c_type in ("token", "clarification"):
                        yield str(chunk.get("content", ""))
                    elif c_type == "error":
                        yield f"\n[Error: {chunk.get('content')}]"
            except Exception as stream_exc:
                yield f"\n[Error: {stream_exc}]"

        return StreamingResponse(token_generator(), media_type="text/plain")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initiate streaming: {exc}",
        ) from exc
