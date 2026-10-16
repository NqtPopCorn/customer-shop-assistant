from fastapi.testclient import TestClient

from customer_shop_assistant.main import app

client = TestClient(app)


def test_api_v1_health() -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_api_v1_chat_clarification_minimal() -> None:
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["clarification_asked"] is True
    assert "chiều cao" in data["reply"]
    assert "cân nặng" in data["reply"]
    assert data["intent"] == "SIZE_ADVICE"
    assert "quick_replies" in data
    assert isinstance(data["quick_replies"], list)
    assert "products" in data


def test_api_chat_alias_compatibility() -> None:
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "conversation_id": "test-alias-123",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test-alias-123"
    assert data["clarification_asked"] is True


def test_api_v1_chat_stream_sse() -> None:
    payload = {
        "message": "Tư vấn size áo giúp mình",
        "clarification_mode": "MINIMAL_CLARIFICATION",
    }
    with client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = [line for line in response.iter_lines() if line.startswith("data: ")]
        assert len(chunks) > 0
