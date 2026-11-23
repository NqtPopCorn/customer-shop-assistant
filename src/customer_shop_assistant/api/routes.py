import base64
import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/v1", tags=["Assistant"])
security = HTTPBearer(auto_error=False)


def extract_user_id_from_jwt(
    credentials: HTTPAuthorizationCredentials | None, secret: str
) -> int | None:
    """Decode Bearer JWT from fashion-shop-be and extract user ID (sub claim)."""
    if not credentials or not credentials.credentials:
        return None
    token = credentials.credentials
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        # Verify HMAC-SHA256 signature if secret configured
        if secret and secret != "change-me":
            signing_input = f"{header_b64}.{payload_b64}".encode()
            expected_sig = hmac.new(
                secret.encode(), signing_input, hashlib.sha256
            ).digest()
            expected_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
            if not hmac.compare_digest(sig_b64.rstrip("="), expected_b64):
                return None

        rem = len(payload_b64) % 4
        padded = payload_b64 + ("=" * (4 - rem) if rem else "")
        payload_bytes = base64.urlsafe_b64decode(padded)
        payload = json.loads(payload_bytes.decode())
        sub = payload.get("sub") or payload.get("id")
        return int(sub) if sub is not None else None
    except Exception:
        return None


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
    auth: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> ChatResponse:
    try:
        token = auth.credentials if auth else None
        user_id = extract_user_id_from_jwt(auth, settings.JWT_SECRET)
        agent = ShopAssistantAgent(settings=settings)
        return await agent.chat(request, user_id=user_id, auth_token=token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate response: {exc}",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    auth: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> StreamingResponse:
    try:
        token = auth.credentials if auth else None
        user_id = extract_user_id_from_jwt(auth, settings.JWT_SECRET)
        agent = ShopAssistantAgent(settings=settings)

        async def token_generator() -> AsyncIterator[str]:
            try:
                async for chunk in agent.chat_stream(
                    request, user_id=user_id, auth_token=token
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as stream_exc:
                err_data = {"type": "error", "content": str(stream_exc)}
                yield f"data: {json.dumps(err_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initiate streaming: {exc}",
        ) from exc


class SessionStateResponse(BaseModel):
    """Debug snapshot of conversation session and graph state."""

    session_id: str
    user_id: int | None = None
    clarification_mode: str
    clarify_phase: str
    gated_turns: int
    active_profile: dict[str, Any] | None = None
    history_turns: int
    context_messages_count: int
    graph_checkpoint_next: list[str] = []


@router.get("/chat/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    session_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SessionStateResponse:
    sess_mgr = get_session_manager(
        ttl_seconds=settings.SESSION_TTL_SECONDS,
        default_max_turns=settings.MAX_SESSION_HISTORY_TURNS,
    )
    session = sess_mgr.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    agent = ShopAssistantAgent(settings=settings)
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_next: list[str] = []
    try:
        graph_state = await agent.graph.aget_state(config)
        if graph_state and graph_state.next:
            checkpoint_next = list(graph_state.next)
    except Exception:
        pass

    return SessionStateResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        clarification_mode=session.clarification_mode,
        clarify_phase=session.clarify_phase.value,
        gated_turns=session.gated_turns,
        active_profile=(
            session.active_profile.model_dump() if session.active_profile else None
        ),
        history_turns=len(session.messages) // 2,
        context_messages_count=len(session.messages),
        graph_checkpoint_next=checkpoint_next,
    )
