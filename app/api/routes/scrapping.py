from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from app.services.knowlage_base import load_info
from app.core.auth import verify_api_key
from app.core.config import settings

router = APIRouter()


@router.post("/update-docs", summary="Обновление данных в базе поиска")
def update_rag(
    authorized: bool = Depends(verify_api_key)
):
    """
    Эндпоинт для обновления данных в базе поиска.
    """

    try:
        load_info(settings.DOC_URL)

        return {"message": "Данные успешно обновлены"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обновлении данных: {str(e)}")