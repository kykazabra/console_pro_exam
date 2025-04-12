from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(..., description="Вопрос пользователя")
    thread_id: Optional[str] = Field(None, description="Идентификатор диалога (опционально)")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Ответ чат-бота")
    thread_id: Optional[str] = Field(None, description="Идентификатор диалога")