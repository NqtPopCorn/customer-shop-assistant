from fastapi.testclient import TestClient

from customer_shop_assistant.main import app

client = TestClient(app)


def test_health_check_returns_200() -> None:
    """Verify that GET /health returns 200 OK and expected JSON schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert data["provider"] in ["ollama", "lmstudio"]
    assert "model" in data
