---
name: add-fastapi-endpoint
description: "Guide for adding a new FastAPI route with Pydantic request/response schemas and Pytest tests."
---

# Add FastAPI Endpoint

Use this procedure when creating or expanding API routes in `customer_shop_assistant`.

## Steps

### 1. Define Request & Response Schemas
Create Pydantic v2 schemas in `src/customer_shop_assistant/schemas/<feature>.py`:
```python
from pydantic import BaseModel, Field


class ItemRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")


class ItemResponse(BaseModel):
    results: list[str] = Field(default_factory=list)
```

### 2. Implement the Route Handler
Create or update router in `src/customer_shop_assistant/api/<feature>.py`:
```python
from fastapi import APIRouter, Depends
from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.<feature> import ItemRequest, ItemResponse

router = APIRouter(prefix="/api/v1/<feature>", tags=["<feature>"])

@router.post("/", response_model=ItemResponse)
async def handle_item(
    request: ItemRequest,
    settings: Settings = Depends(get_settings),
) -> ItemResponse:
    return ItemResponse(results=[])
```

### 3. Register Router in `main.py`
In `src/customer_shop_assistant/main.py`:
```python
from customer_shop_assistant.api.<feature> import router as feature_router

app.include_router(feature_router)
```

### 4. Write an Automated Test
In `tests/test_<feature>.py`:
```python
from fastapi.testclient import TestClient
from customer_shop_assistant.main import app

client = TestClient(app)


def test_feature_endpoint() -> None:
    response = client.post("/api/v1/<feature>/", json={"query": "test"})
    assert response.status_code == 200
    assert "results" in response.json()
```

### 5. Verification
Run Ruff and Pytest to verify correctness:
```powershell
uv run ruff check .
uv run pytest tests/test_<feature>.py
```
