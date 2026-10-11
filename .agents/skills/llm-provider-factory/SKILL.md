---
name: llm-provider-factory
description: "Procedure for constructing and configuring LangChain chat models for Ollama and LM Studio."
---

# LLM Provider Factory

Use this skill when integrating or modifying chat model generation via `customer_shop_assistant.llm.factory`.

## Steps

### 1. Inspect Environment & Settings
Verify the current provider selection in `.env` or settings:
- `LLM_PROVIDER`: `"ollama"` or `"lmstudio"`
- Check provider URLs and model names:
  - Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
  - LM Studio: `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL`, `LMSTUDIO_API_KEY`
- Inference parameters: `TEMPERATURE`, `TIMEOUT`, `SEED`.

### 2. Instantiate Model via Factory
Always call `get_chat_model`:
```python
from customer_shop_assistant.config import get_settings
from customer_shop_assistant.llm.factory import get_chat_model

settings = get_settings()
model = get_chat_model(settings)
```

### 3. Invoke Chat Model
Build standard LangChain messages:
```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful clothing shop assistant."),
    HumanMessage(content="Do you have black jackets?"),
]
response = await model.ainvoke(messages)
print(response.content)
```

### 4. Switch Providers for Testing
To test with Ollama:
```powershell
$env:LLM_PROVIDER="ollama"
uv run python -c "from customer_shop_assistant.llm.factory import get_chat_model; print(get_chat_model())"
```
To test with LM Studio:
```powershell
$env:LLM_PROVIDER="lmstudio"
uv run python -c "from customer_shop_assistant.llm.factory import get_chat_model; print(get_chat_model())"
```
