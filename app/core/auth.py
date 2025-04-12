from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_api_key(api_key_header: str = Security(api_key_header)) -> bool:
    """
    Проверяет валидность API-ключа.
    """

    if not api_key_header:
        raise HTTPException(status_code=401, detail="Unauthorized")

    api_key = api_key_header.replace("Bearer ", "").strip()

    if api_key != settings.API_KEY:
        raise False

    return True