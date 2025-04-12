import pytest
from fastapi.testclient import TestClient
from app.core.config import settings


def test_health_check(client: TestClient):
    """
    Тест проверки работоспособности сервиса.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Service is running"}


def test_chat_endpoint_unauthorized(client: TestClient):
    """
    Тест проверки аутентификации.
    """
    response = client.post("/chat", json={"question": "Как оформить контракт?"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_chat_endpoint_success(client: TestClient):
    """
    Тест успешного запроса к /chat.
    """
    headers = {"Authorization": f"Bearer {settings.API_KEY}"}
    data = {"question": "Как оформить контракт?"}
    response = client.post("/chat", json=data, headers=headers)
    assert response.status_code == 200
    assert "answer" in response.json()