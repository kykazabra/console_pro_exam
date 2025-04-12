from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import process_user_query
from app.core.auth import verify_api_key


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    authorized: bool = Depends(verify_api_key)
):
    try:
        result = process_user_query(
            question=request.question,
            thread_id=request.thread_id
        )

        return ChatResponse(
            answer=result["answer"],
            thread_id=result.get("thread_id")
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")