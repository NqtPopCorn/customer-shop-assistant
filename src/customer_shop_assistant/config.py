from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and LLM settings loaded from environment or .env file."""

    APP_NAME: str = "Customer Shop Assistant"
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    CORS_ORIGINS: list[str] = [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ]

    LLM_PROVIDER: Literal["ollama", "lmstudio"] = "lmstudio"
    TEMPERATURE: float = 0.0
    SEED: int | None = None
    TIMEOUT: float = 60.0

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_URL"),
        description="Ollama server endpoint",
    )
    OLLAMA_MODEL: str = "llama3.2"
    EMBEDDING_PROVIDER: Literal["ollama", "tfidf"] = "ollama"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    LMSTUDIO_BASE_URL: str = Field(
        default="http://127.0.0.1:1235/v1",
        validation_alias=AliasChoices("LMSTUDIO_BASE_URL", "LMSTUDIO_URL"),
        description="LM Studio OpenAI-compatible endpoint",
    )
    LMSTUDIO_MODEL: str = "google/gemma-4-e2b"
    LMSTUDIO_API_KEY: str = "lm-studio"

    # Main E-commerce Backend API Settings (fashion-shop-be)
    MAIN_BE_BASE_URL: str = Field(
        default="http://localhost:3000/api",
        validation_alias=AliasChoices(
            "MAIN_BE_BASE_URL",
            "SHOP_API_URL",
            "BACKEND_API_URL",
            "API_BASE_URL",
        ),
        description="Base URL of the main e-commerce backend REST API",
    )
    MAIN_BE_TIMEOUT: float = Field(
        default=5.0,
        validation_alias=AliasChoices("MAIN_BE_TIMEOUT", "SHOP_API_TIMEOUT"),
        description="HTTP timeout in seconds when calling the backend API",
    )

    # JWT Authentication & Session Memory Settings
    JWT_SECRET: str = Field(
        default="change-me",
        description="JWT secret key for user auth tokens from fashion-shop-be",
    )
    SESSION_TTL_SECONDS: int = Field(
        default=3600,
        description="In-memory chat session expiration in seconds (default 60m)",
    )
    MAX_SESSION_HISTORY_TURNS: int = Field(
        default=8,
        description="Sliding window conversation turns for prompt context",
    )
    DEFAULT_CLARIFICATION_MODE: str = Field(
        default="CONDITIONAL_CLARIFICATION",
        description="Default clarification strategy when not provided by client",
    )
    DEBUG_CHAT_META: bool = Field(
        default=False,
        description="Include debug metadata (mode, phase, tools) in ChatResponse",
    )

    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
