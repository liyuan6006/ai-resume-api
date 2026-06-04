from fastapi import APIRouter

from pydantic import BaseModel

from rag_service import RagService


router = APIRouter()

rag_service = RagService()


class AskRequest(BaseModel):

    question: str

@router.post("/ask")
async def ask_question(
    request: AskRequest
):

    answer = rag_service.ask(
        request.question
    )

    return {
        "answer": answer
    }