from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Customer Shop Assistant"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # LM Studio Settings
    lmstudio_base_url: str = "http://127.0.0.1:1235/v1"
    lmstudio_model: str = "google/gemma-4-e2b"
    lmstudio_api_key: str = "lm-studio"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
