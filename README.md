# Customer Shop Assistant

Research prototype AI chatbot for a clothing store, designed to evaluate and compare local LLM models (Ollama vs. LM Studio) for customer inquiries, sizing advice, product catalog questions, and store policies.

---

## Tech Stack
- **Language**: Python `>= 3.14`
- **Package & Environment Manager**: [uv](https://docs.astral.sh/uv/)
- **Backend**: FastAPI, Uvicorn
- **AI Framework**: LangChain (`langchain`, `langchain-ollama`, `langchain-openai`)
- **Settings**: `pydantic-settings`
- **Testing & Tooling**: Pytest, HTTPX, Ruff

---

## Directory Structure
```text
customer-shop-assistant/
├── AGENTS.md                         # Single source of truth for coding agents
├── README.md                         # Project documentation
├── pyproject.toml                    # UV metadata, dependencies & tool configs
├── .env.example                      # Environment variables reference
├── .agents/                          # Antigravity agent rules, skills & workflows
├── data/                             # Mock products and store policies
├── eval/                             # Evaluation dataset (Vietnamese customer queries)
├── logs/                             # Research telemetry logs (JSONL, git-ignored)
├── tests/                            # Automated test suite
├── _legacy/                          # Archived pre-scaffolding code
└── src/customer_shop_assistant/
    ├── __init__.py                   # Package exports
    ├── config.py                     # Pydantic settings & get_settings()
    ├── main.py                       # FastAPI application & GET /health
    ├── api/                          # Route handlers
    ├── services/                     # Business logic
    ├── schemas/                      # Pydantic schemas
    ├── loaders/                      # Catalog and policy loaders
    └── llm/
        ├── __init__.py
        └── factory.py                # LLM model factory
```

---

## Getting Started

### 1. Install & Sync Dependencies
```powershell
uv sync
```

### 2. Configure Environment
Copy `.env.example` to `.env` (if not already created):
```powershell
Copy-Item .env.example .env
```

### 3. Switch LLM Provider
Set `LLM_PROVIDER` in your `.env` file or environment:
- **LM Studio (Default)**:
  ```dotenv
  LLM_PROVIDER="lmstudio"
  LMSTUDIO_BASE_URL="http://127.0.0.1:1235/v1"
  LMSTUDIO_MODEL="google/gemma-4-e2b"
  LMSTUDIO_API_KEY="lm-studio"
  ```
- **Ollama**:
  ```dotenv
  LLM_PROVIDER="ollama"
  OLLAMA_BASE_URL="http://localhost:11434"
  OLLAMA_MODEL="llama3.2"
  ```

### 4. Run the Application
Start the development server with Uvicorn:
```powershell
uv run uvicorn customer_shop_assistant.main:app --reload --host 127.0.0.1 --port 8000
```
Or use the registered script:
```powershell
uv run customer-shop-assistant
```

### 5. Check Health Status
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```
Expected response:
```json
{
  "status": "healthy",
  "app_name": "Customer Shop Assistant",
  "provider": "lmstudio",
  "model": "google/gemma-4-e2b"
}
```

---

## Development & Testing Commands
```powershell
# Run unit tests
uv run pytest

# Check code formatting and linting
uv run ruff check .

# Automatically format code
uv run ruff format .
```
