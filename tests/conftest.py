import pytest
from app import app as fastapi_app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Фикстура для тестирования API.
    """
    with TestClient(fastapi_app) as client:
        yield client
