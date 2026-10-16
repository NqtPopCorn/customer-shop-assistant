from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from customer_shop_assistant.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Instantiate and return a LangChain ChatModel based on application settings.

    Supports 'ollama' and 'lmstudio' (OpenAI-compatible) providers.
    """
    s = settings or get_settings()

    extra_kwargs: dict[str, Any] = {}
    if s.SEED is not None:
        extra_kwargs["seed"] = s.SEED

    if s.LLM_PROVIDER == "ollama":
        return ChatOllama(
            base_url=s.OLLAMA_BASE_URL,
            model=s.OLLAMA_MODEL,
            temperature=s.TEMPERATURE,
            timeout=s.TIMEOUT,
            streaming=True,
            **extra_kwargs,
        )

    return ChatOpenAI(
        base_url=s.LMSTUDIO_BASE_URL,
        api_key=s.LMSTUDIO_API_KEY,
        model=s.LMSTUDIO_MODEL,
        temperature=s.TEMPERATURE,
        timeout=s.TIMEOUT,
        streaming=True,
        **extra_kwargs,
    )
