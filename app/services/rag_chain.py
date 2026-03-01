"""RAG chain composition (LangChain-style glue).

This module demonstrates how to wire together PDF/audio extractors, the vector store,
search, and the LLM engine to answer user queries.
"""
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.vector_store import init_vector_store, persist_vectors
from app.services.search_service import bm25_search
from app.services.llm_engine import generate_answer_from_model


def run_rag(query: str, top_k: int = 5) -> str:
    """Very small RAG pipeline stub that demonstrates flow."""
    # 1) retrieve candidates (stub)
    docs = bm25_search(query, top_k=top_k)

    # 2) construct context (simple join)
    context = "\n\n".join(docs) if docs else None

    # 3) generate answer
    answer = generate_answer_from_model(query, context=[context] if context else None)
    return answer
