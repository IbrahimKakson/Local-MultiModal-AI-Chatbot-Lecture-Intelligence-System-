"""LLM engine integration (LlamaCpp / Phi-3 Mini) - renamed from llm_service.
"""
import os
import multiprocessing
from app.core.config import settings
from app.core.prompts import LLAMA3_RAG_TEMPLATE

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Global instance
_llm_instance = None

def unload_model():
    """Wipe the model from memory to free RAM."""
    global _llm_instance
    if _llm_instance is not None:
        print("Unloading LLM from RAM...")
        del _llm_instance
        _llm_instance = None
        import gc
        gc.collect()

def load_model(model_name: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"):
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
        n_ctx=2048,  # Reduced context window to save RAM
        n_threads=max(1, multiprocessing.cpu_count() // 2), # Optimal physical cores
        n_gpu_layers=0, # Fallback baseline (0 = CPU only)
        verbose=False 
    )
    return _llm_instance

def generate_answer_from_model(query: str, context: list[str] | None = None) -> str:
    """Generate an answer from the locally loaded model."""
    llm = load_model()

    if context and any(context):
        context_str = "\n\n".join(context)
        prompt = LLAMA3_RAG_TEMPLATE.format(context=context_str, question=query)
    else:
        # Pure conversational prompt for short greetings without strict RAG rules
        prompt = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful and friendly tutor assistant. Speak naturally.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Standard completion settings
    response = llm(
        prompt,
        max_tokens=1024, # Support longer paragraph responses
        temperature=0.7,
        top_p=0.95,
        stop=["<|eot_id|>", "<|start_header_id|>"],
    )

    # Extract the string response from LlamaCpp's verbose dictionary
    return response["choices"][0]["text"].strip()


def stream_answer_from_model(query: str, context: list[str] | None = None):
    """Stream tokens one-by-one from the locally loaded model.

    Yields individual text tokens as they are generated (for real-time UI).
    """
    llm = load_model()

    if context and any(context):
        context_str = "\n\n".join(context)
        prompt = LLAMA3_RAG_TEMPLATE.format(context=context_str, question=query)
    else:
        prompt = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful and friendly tutor assistant. Speak naturally.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    for token in llm(
        prompt,
        max_tokens=1024, # Support longer paragraph responses
        temperature=0.7,
        top_p=0.95,
        stop=["<|eot_id|>", "<|start_header_id|>"],
        stream=True,
    ):
        text = token["choices"][0]["text"]
        if text:
            yield text
