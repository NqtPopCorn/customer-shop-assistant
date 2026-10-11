---
trigger: model_decision
description: "Guidelines for working with LLM providers, chat models, and the provider factory."
---

# LLM Provider Architecture Rules

## 1. Single Entry Point Factory
- All LangChain chat model instances must be obtained exclusively through `customer_shop_assistant.llm.factory.get_chat_model(settings)`.
- Never directly instantiate `ChatOllama` or `ChatOpenAI` in router endpoints or service classes.

## 2. Provider Interchangeability
- The system must seamlessly support both `ollama` and `lmstudio` as defined by `settings.LLM_PROVIDER`:
  - `ollama`: Uses `ChatOllama` with `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
  - `lmstudio`: Uses `ChatOpenAI` (OpenAI-compatible) with `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL`, and `LMSTUDIO_API_KEY`.
- Both providers must produce consistent output formats using standard LangChain message structures (`HumanMessage`, `AIMessage`, `SystemMessage`).

## 3. Parameter Propagation
- Always forward runtime inference parameters:
  - `temperature`: Configured via `settings.TEMPERATURE`.
  - `timeout`: Configured via `settings.TIMEOUT`.
  - `seed`: Configured via `settings.SEED` (passed when not `None`).

## 4. Zero Hardcoded Credentials
- Never hardcode model tags, base URLs, or API keys in code.
- Read all connection details from `Settings` loaded by `pydantic-settings`.
