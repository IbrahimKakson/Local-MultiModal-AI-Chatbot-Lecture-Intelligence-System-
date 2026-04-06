"""LLM engine integration (LlamaCpp / Mistral) - renamed from llm_service.
"""
import os
from app.core.config import settings
from app.core.prompts import MISTRAL_RAG_TEMPLATE

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Global instance
_llm_instance = None

def load_model(model_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    """Load the local GGUF model using LlamaCpp for CPU execution."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if Llama is None:
        raise ImportError("llama-cpp-python is not installed. Please install it to use the local LLM.")

    model_path = os.path.join(settings.models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download it.")

    # Optimized for CPU usage minimum requirements
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=4096,  # 4K context window to balance RAM vs Capability
        n_threads=None, # None auto-detects optimal CPU thread count
        n_gpu_layers=0, # Fallback baseline (0 = CPU only)
        verbose=False 
    )
    return _llm_instance

def generate_answer_from_model(query: str, context: list[str] | None = None) -> str:
    """Generate an answer from the locally loaded Mistral model."""
    llm = load_model()

    # Format the context tightly
    context_str = "No relevant context found."
    if context:
        context_str = "\n\n".join(context)

    # Inject into the Mistral Template
    prompt = MISTRAL_RAG_TEMPLATE.format(context=context_str, question=query)

    # Standard completion settings
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        stop=["[/INST]", "user:", "User:"] # Safety stops so mistral doesn't talk to itself
    )

    # Extract the string response from LlamaCpp's verbose dictionary
    return response["choices"][0]["text"].strip()
