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

    # ask() now returns {"answer": str, "sources": [...]} so the response
    # carries citation metadata alongside the answer.
    return rag_service.ask(
        request.question
    )