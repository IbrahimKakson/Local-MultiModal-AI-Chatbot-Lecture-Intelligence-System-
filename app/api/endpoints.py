from fastapi import APIRouter, UploadFile, File
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Accept a chat request and (stub) return an answer."""
    # TODO: wire into services.llm_service.generate_answer
    return {"answer": "This is a placeholder answer."}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a file upload and store it under data/uploads."""
    upload_path = f"data/uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "saved_to": upload_path}
