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
