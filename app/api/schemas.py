from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
