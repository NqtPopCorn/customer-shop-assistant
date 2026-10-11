# Workflow: /new-feature

Standard 4-step workflow for implementing a new feature in `customer-shop-assistant`.

## Steps

1. **Plan**
   - Review requirements and inspect existing modules (`config.py`, `schemas/`, `api/`, `services/`, `loaders/`).
   - Create or update an implementation plan outlining changes and expected tests.
   - Confirm alignment before writing code.

2. **Implement**
   - Write typed Pydantic models in `src/customer_shop_assistant/schemas/`.
   - Implement business logic in `src/customer_shop_assistant/services/`.
   - Expose async endpoint in `src/customer_shop_assistant/api/` and register in `main.py`.
   - Ensure settings are accessed solely via `get_settings()`.

3. **Test**
   - Add unit/integration tests in `tests/`.
   - Run tests using `uv run pytest`.
   - Ensure all test assertions pass.

4. **Lint & Format**
   - Run `uv run ruff check .` and fix any lint issues.
   - Run `uv run ruff format .` to format files.
