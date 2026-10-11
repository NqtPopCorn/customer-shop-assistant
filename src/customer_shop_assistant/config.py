from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and LLM settings loaded from environment or .env file."""

    APP_NAME: str = "Customer Shop Assistant"
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    LLM_PROVIDER: Literal["ollama", "lmstudio"] = "lmstudio"
    TEMPERATURE: float = 0.0
    SEED: int | None = None
    TIMEOUT: float = 60.0

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1235/v1"
    LMSTUDIO_MODEL: str = "google/gemma-4-e2b"
    LMSTUDIO_API_KEY: str = "lm-studio"

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
