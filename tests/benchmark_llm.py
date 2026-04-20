"""Benchmark for the Local LLM Engine (Week 7)."""
import time
from app.services.llm_engine import load_model, generate_answer_from_model

def run_benchmark():
    print("Loading Local LLM into memory (CPU)... This may take 5-10 seconds.")
    start_load = time.time()
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        print("Please ensure you have downloaded 'Phi-3-mini-4k-instruct-q4.gguf' to data/models/")
        return
    except ImportError as e:
        print(f"FAILED: {e}")
        return

    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f} seconds.\n")

    print("Running a test RAG query...")
    context = ["The capital of France is Paris. It is known for the Eiffel Tower."]
    query = "What is the capital of France and what is it known for?"
    
    start_infer = time.time()
    answer = generate_answer_from_model(query=query, context=context)
    infer_time = time.time() - start_infer

    print("================ Response ================")
    print(answer)
    print("==========================================")
    print(f"Inference generated in {infer_time:.2f} seconds.")

if __name__ == "__main__":
    run_benchmark()
