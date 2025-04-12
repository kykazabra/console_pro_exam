from fastapi import FastAPI
from .api.routes.chat import router as chat_router
from .api.routes.scrapping import router as scrapping_router
from .core.config import settings


def create_app() -> FastAPI:
    """
    Создает экземпляр FastAPI приложения.
    """
    app = FastAPI(
        title="ChatBot Agent RAG API",
        description="API сервис чат-бота поддержки с использованием RAG и LangGraph",
        version="1.0.0"
    )

    app.include_router(chat_router)
    app.include_router(scrapping_router)

    @app.get("/health")
    def health_check():
        """
        Проверка работоспособности сервиса.
        """
        return {"status": "ok", "message": "Service is running"}

    return app


app = create_app()