import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.main import app

client = TestClient(app)


def test_api_v1_health() -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_api_v1_chat_clarification_minimal() -> None:
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=(
                        "Dạ để tư vấn size chuẩn xác nhất, bạn vui lòng cho mình "
                        "biết chiều cao và cân nặng của bạn nhé?"
                    )
                )
            ]
        )
    )
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    with patch(
        "customer_shop_assistant.services.agent_service.get_chat_model",
        return_value=fake_model,
    ):
        res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["clarification_asked"] is True
    assert "chiều cao" in data["reply"]
    assert "cân nặng" in data["reply"]
    assert "quick_replies" not in data
    assert "products" in data


def test_api_chat_alias_compatibility() -> None:
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Dạ bạn vui lòng cho mình biết chiều cao và cân nặng nhé?"
                )
            ]
        )
    )
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "conversation_id": "test-alias-123",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    with patch(
        "customer_shop_assistant.services.agent_service.get_chat_model",
        return_value=fake_model,
    ):
        res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test-alias-123"
    assert data["clarification_asked"] is True


def test_api_v1_chat_stream_sse() -> None:
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Dạ bạn vui lòng cho mình biết chiều cao và cân nặng nhé?"
                )
            ]
        )
    )
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    with patch(
        "customer_shop_assistant.services.agent_service.get_chat_model",
        return_value=fake_model,
    ):
        with client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            chunks = [
                line for line in response.iter_lines() if line.startswith("data: ")
            ]
            assert len(chunks) > 0


def test_api_chat_session_continuity_and_jwt() -> None:
    # Create dummy unsigned/mock JWT with sub=42
    header_b64 = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode())
        .decode()
        .rstrip("=")
    )
    payload_b64 = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": 42, "email": "test@example.com"}).encode()
        )
        .decode()
        .rstrip("=")
    )
    fake_jwt = f"{header_b64}.{payload_b64}.fake_sig"

    session_id = "test-session-persist-42"
    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Dạ với số đo này bạn mặc size L nhé!")])
    )
    with patch(
        "customer_shop_assistant.services.agent_service.get_chat_model",
        return_value=fake_model,
    ):
        res1 = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {fake_jwt}"},
            json={
                "session_id": session_id,
                "message": "Mình cao 1m72 nặng 68kg, tư vấn áo sơ mi công sở",
                "user_profile": {
                    "height_cm": 172.0,
                    "weight_kg": 68.0,
                },
            },
        )
        assert res1.status_code == 200

    # Verify session manager stored the active profile
    from customer_shop_assistant.services.session_manager import (
        get_session_manager,
    )

    mgr = get_session_manager()
    session = mgr.get_session(session_id)
    assert session is not None
    assert session.active_profile is not None
    assert session.active_profile.height_cm == 172.0
    assert session.active_profile.weight_kg == 68.0
    assert session.user_id == 42

    # Verify GET /api/v1/chat/sessions/{session_id}/state
    state_res = client.get(f"/api/v1/chat/sessions/{session_id}/state")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["session_id"] == session_id
    assert state_data["user_id"] == 42
    assert state_data["active_profile"]["height_cm"] == 172.0
    assert state_data["clarify_phase"] in ("INTERVIEW", "OPEN")

    # Verify 404 for non-existent session
    not_found_res = client.get("/api/v1/chat/sessions/non-existent-9999/state")
    assert not_found_res.status_code == 404


def test_api_v1_chat_debug_meta() -> None:
    session_id = "test-debug-meta-session"
    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Chào bạn! Shop có thể tư vấn gì cho bạn?")])
    )
    with patch(
        "customer_shop_assistant.services.agent_service.get_chat_model",
        return_value=fake_model,
    ):
        res = client.post(
            "/api/v1/chat",
            json={
                "session_id": session_id,
                "message": "Gợi ý váy đi tiệc",
                "clarification_mode": "CONDITIONAL_CLARIFICATION",
            },
        )
    assert res.status_code == 200
    data = res.json()
    assert "debug_meta" in data
    assert data["debug_meta"] is not None
    d_meta = data["debug_meta"]
    assert "mode" in d_meta
    assert "phase" in d_meta
    assert "runs" in d_meta
    assert isinstance(d_meta["runs"], list)
    assert "state_snapshot" in d_meta
    assert d_meta["state_snapshot"]["detected_slots"]["known_occasion"] == "party"
