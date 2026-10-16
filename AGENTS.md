# AGENTS.md — Customer Shop Assistant

Single source of truth for coding agents working on `customer-shop-assistant`.

## 1. Project Overview & Research Goal
- **Project**: `customer-shop-assistant`, a research prototype chatbot for an e-commerce clothing store.
- **Reference Catalog Source**: `H:\Code\PET\e-commerce-clothes` (clothing store domain reference).
- **Product Requirements Document**: [docs/plans/prd.md](file:///h:/Code/PET/customer-shop-assistant/docs/plans/prd.md) (PRD v1.0 specifications & decisions).
- **Research Goal**: Study how an AI chatbot answers clothing store customer questions (sizes, recommendations, inventory, store policies, clarification modes).
- **Core Philosophy**: Speed, clarity, and rapid experimentation matter more than production polish. Do NOT over-engineer.

## 2. Tech Stack & Commands
- **Runtime**: Python `>= 3.14`, package-managed by **uv** (`src/` layout with `uv_build`).
- **Core Frameworks**: FastAPI, Uvicorn, LangChain (`langchain`, `langchain-ollama`, `langchain-openai`), `pydantic-settings`.
- **Tooling**: Ruff (linting/formatting), Pytest (testing), HTTPX (testing client).
- **LLM Providers**: Ollama (`llama3.2`) and LM Studio (`google/gemma-4-e2b`, OpenAI-compatible).
- **Embedding Providers**: Ollama (`nomic-embed-text`).

### Exact Terminal Commands
```powershell
uv sync                                                   # Sync dependencies
uv run uvicorn customer_shop_assistant.main:app --reload  # Start dev server
uv run pytest                                             # Run unit tests
uv run ruff check .                                       # Lint checks
uv run ruff format .                                      # Format codebase
```

## 3. Directory Map
```text
customer-shop-assistant/
├── AGENTS.md                         # Single source of truth (this file)
├── README.md                         # Human-facing guide & setup
├── pyproject.toml                    # Project metadata, dependencies, tool configs
├── .env.example                      # Template for environment variables
├── docs/                             # Documentation, plans & requirements
│   ├── plans/
│   │   ├── prd.md                    # PRD v1.0 (Product Requirements Document)
│   │   └── macro.md                  # Macro roadmap & phase dependencies
│   └── requirements/                 # Business requirements
├── .agents/                          # Antigravity Workspace Customizations
│   ├── rules/                        # python-style, llm-provider, research-logging
│   ├── skills/                       # add-fastapi-endpoint, llm-provider-factory, etc.
│   └── workflows/                    # /new-feature, /run-demo, /compare-models
├── data/                             # Mock data: products.json, policies.json
├── eval/                             # Evaluation datasets: questions.json
├── logs/                             # Ignored telemetry logs (chat_turns.jsonl)
├── tests/                            # Pytest test suite (test_health.py)
└── src/customer_shop_assistant/
    ├── __init__.py                   # Exports app and main()
    ├── config.py                     # Settings class with get_settings()
    ├── main.py                       # FastAPI app and GET /health
    ├── api/                          # Future route handlers
    ├── services/                     # Future business logic
    ├── schemas/                      # Future Pydantic request/response models
    ├── loaders/                      # Future catalog/policy data loaders
    └── llm/
        ├── __init__.py               # LLM subpackage
        └── factory.py                # get_chat_model(settings) provider factory
```

## 4. Coding Conventions
- **Strict Typing**: Type hints on every function parameter, return type, and class attribute.
- **Async Endpoints**: FastAPI endpoints must be `async def`.
- **Pydantic v2**: Use `BaseModel` with explicit field definitions (`Field(...)`) and `model_dump()`.
- **Settings Management**: Access settings exclusively through `get_settings()` from `customer_shop_assistant.config`. Never instantiate module-level singletons or read `os.environ` directly in feature code.
- **Modularity**: Small, single-purpose modules. Keep business logic in `services/`, routes in `api/`, models in `schemas/`.
- **Zero Hardcoding**: Never hardcode model identifiers, base URLs, or secrets in code.

## 5. Golden Rules for Agents
1. **Never commit `.env`**: Secret keys and local endpoints belong in `.env` only.
2. **Never edit `uv.lock` manually**: Manage dependencies only with `uv add` or `uv remove`.
3. **Ask before adding dependencies**: Always propose new dependencies to the user before running `uv add`.
4. **Interchangeable Providers**: Keep Ollama and LM Studio swappable via `LLM_PROVIDER` in settings.
5. **Keep It Simple**: This is a research prototype; avoid premature microservices, excessive layers, or unneeded abstractions.

## 6. Definition of Done
A task is DONE only when:
1. All changes run cleanly without runtime errors.
2. `uv run ruff check .` passes with zero lint errors.
3. `uv run pytest` passes all tests.
4. Relevant documentation or rules are kept in sync.