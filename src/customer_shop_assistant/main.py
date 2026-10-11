from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from customer_shop_assistant.api.routes import router as api_router
from customer_shop_assistant.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Customer Shop Assistant backend API powered by FastAPI and Ollama.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name} API!",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def main() -> None:
    uvicorn.run(
        "customer_shop_assistant.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
