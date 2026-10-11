from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI

from customer_shop_assistant.config import Settings, get_settings

app = FastAPI(
    title="Customer Shop Assistant",
    description="Research prototype chatbot for clothing customer shop assistant",
    version="0.1.0",
)


@app.get("/health")
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Health check endpoint reporting service status and active LLM configuration."""
    active_model = (
        settings.OLLAMA_MODEL
        if settings.LLM_PROVIDER == "ollama"
        else settings.LMSTUDIO_MODEL
    )
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "provider": settings.LLM_PROVIDER,
        "model": active_model,
    }


def main() -> None:
    """Entry point for running uvicorn server via CLI script."""
    settings = get_settings()
    uvicorn.run(
        "customer_shop_assistant.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
