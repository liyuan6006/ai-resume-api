import os

from fastapi import APIRouter
from fastapi import UploadFile
from ingestion_service import (
    IngestionService
)
router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)
ingestion_service = IngestionService()


@router.post("/upload")
async def upload_document(
    file: UploadFile
):

    path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(path, "wb") as f:
        f.write(
            await file.read()
        )

    chunks = ingestion_service.ingest_file(
        path
    )
        
    return {
        "message": "uploaded",
        "file": file.filename,
         "chunks": chunks
    }