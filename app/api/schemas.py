from pydantic import BaseModel
from typing import List, Optional

# --- Chat Models (Week 1 / Existing) ---
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str

# --- Document Processing Models (Week 2) ---
class DocumentChunk(BaseModel):
    text: str
    page_number: int
    source: str
    metadata: Optional[dict] = {}

class PDFResponse(BaseModel):
    filename: str
    chunks: List[DocumentChunk]
    total_chunks: int
