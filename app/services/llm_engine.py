"""LLM engine integration (LlamaCpp / Mistral) - renamed from llm_service.
"""


def load_model(path: str):
    """Stub: load model from path (mistral, phi-2, etc.)"""
    return None


def generate_answer_from_model(query: str, context: list[str] | None = None) -> str:
    """Stub: generate an answer from the model."""
    return "Generated answer (stub from llm_engine)."
