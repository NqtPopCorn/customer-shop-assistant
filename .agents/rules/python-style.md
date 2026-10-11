---
trigger: always_on
description: "Python code style, type annotations, and FastAPI best practices."
---

# Python Code Style Rules

## 1. Type Annotations
- Every function, method parameter, and return value must have explicit type annotations.
- Use standard library types (`int | None`, `list[str]`, `dict[str, Any]`) over legacy `typing` constructs where possible.
- Avoid using `Any` unless interfacing with untyped third-party payloads.

## 2. FastAPI & Async
- All FastAPI router endpoints must be defined using `async def`.
- Return types must be annotated with either a Pydantic model, `Response`, or explicit typed primitive/dictionary.
- Inject dependencies using FastAPI `Depends(...)`.

## 3. Pydantic v2 Models
- Inherit from `pydantic.BaseModel`.
- Use `Field(..., description="...")` with validation constraints (e.g. `min_length`, `ge`, `le`).
- Serialize using `model.model_dump()` or `model.model_dump_json()`. Do not use deprecated `.dict()` or `.json()`.

## 4. Configuration & Settings
- Import settings via `from customer_shop_assistant.config import get_settings`.
- Call `settings = get_settings()` inside endpoint/service functions or FastAPI dependencies.
- Never instantiate module-level singleton objects that read settings during import time.

## 5. Code Cleanliness & Formatting
- All code must conform to Ruff rules configured in `pyproject.toml`.
- Run `uv run ruff check .` and `uv run ruff format .` before marking any task as complete.
- Maximum line length is 88 characters.
