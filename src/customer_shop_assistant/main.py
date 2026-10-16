import argparse
import logging
import sys
import time
from typing import Annotated

import anyio
import uvicorn
from fastapi import Depends, FastAPI, Request, Response

from customer_shop_assistant.api.routes import router as api_router
from customer_shop_assistant.cli import run_cli
from customer_shop_assistant.config import Settings, get_settings

logger = logging.getLogger("customer_shop_assistant.api")

app = FastAPI(
    title="Customer Shop Assistant",
    description="Research prototype chatbot for clothing customer shop assistant",
    version="0.1.0",
)
app.include_router(api_router)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next) -> Response:
    """Log HTTP request method, endpoint, status code, and latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    client_ip = request.client.host if request.client else "-"
    logger.info(
        "%s %s %d - %.1fms - %s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client_ip,
    )
    return response


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
    """Entry point for running server or CLI session."""
    parser = argparse.ArgumentParser(description="Customer Shop Assistant")
    parser.add_argument(
        "--cli", action="store_true", help="Launch interactive CLI chat session"
    )
    args = parser.parse_args()

    if args.cli:
        anyio.run(run_cli)
        sys.exit(0)

    settings = get_settings()
    uvicorn.run(
        "customer_shop_assistant.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
