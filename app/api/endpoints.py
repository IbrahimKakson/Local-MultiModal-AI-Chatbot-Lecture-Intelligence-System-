from fastapi import APIRouter, UploadFile, File
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Accept a chat request and (stub) return an answer."""
    # TODO: wire into services.llm_service.generate_answer
    return {"answer": "This is a placeholder answer."}

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a file upload, save it, and process it based on type."""
    os.makedirs("data/uploads", exist_ok=True)
    upload_path = f"data/uploads/{file.filename}"
    
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)
        
    if file.filename.lower().endswith(".pdf"):
        pdf_service = PDFService()
        chunks = pdf_service.extract_text_from_pdf(content, file.filename)
        return {"filename": file.filename, "type": "pdf", "chunks_extracted": len(chunks)}
        
    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
        audio_service = AudioService()
        segments = audio_service.transcribe_audio(upload_path)
        return {"filename": file.filename, "type": "audio", "segments_extracted": len(segments)}
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
