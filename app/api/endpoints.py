import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.schemas import ChatRequest, ChatResponse
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Accept a chat request and return a RAG-powered answer."""
    from app.services.rag_chain import RAGChain

    rag = RAGChain(top_k=req.top_k)
    result = rag.ask(req.query)
    return {"answer": result["answer"]}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a file upload, process it, and store chunks in the vector database."""
    os.makedirs("data/uploads", exist_ok=True)
    upload_path = f"data/uploads/{file.filename}"

    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    vector_store = VectorStoreService()

    if file.filename.lower().endswith(".pdf"):
        pdf_service = PDFService()
        chunks = pdf_service.extract_text_from_pdf(content, file.filename)

        # Store chunks in vector database with source metadata
        if chunks:
            texts = [c.text for c in chunks]
            metadatas = [{"source": file.filename, "page": i + 1} for i in range(len(chunks))]
            ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
            vector_store.add_documents(texts, metadatas, ids)

        return {"filename": file.filename, "type": "pdf", "chunks_extracted": len(chunks)}

    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
        audio_service = AudioService()
        segments = audio_service.transcribe_audio(upload_path)

        # Store transcribed segments in vector database with timestamp metadata
        if segments:
            texts = [seg["text"] for seg in segments]
            metadatas = [
                {"source": file.filename, "start": seg["start"], "end": seg["end"]}
                for seg in segments
            ]
            ids = [f"{file.filename}_seg_{i}" for i in range(len(segments))]
            vector_store.add_documents(texts, metadatas, ids)

        return {"filename": file.filename, "type": "audio", "segments_extracted": len(segments)}

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
