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
        print("Please ensure you have downloaded 'Llama-3.2-1B-Instruct-Q4_K_M.gguf' to data/models/")
        return
    except ImportError as e:
        print(f"FAILED: {e}")
        return

    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f} seconds.\n")

    print("Running a test RAG query (with Performance Tracking)...")
    context = ["The capital of France is Paris. It is known for the Eiffel Tower."]
    query = "What is the capital of France and what is it known for?"
    
    # Apply the decorator locally for the test
    from tests.performance_decorator import measure_generation_performance
    from app.services.llm_engine import stream_answer_from_model
    
    tracked_stream = measure_generation_performance(stream_answer_from_model)
    
    print("================ Response ================")
    answer = ""
    for token in tracked_stream(query=query, context=context):
        print(token, end="", flush=True)
        answer += token
    print("\n==========================================")

if __name__ == "__main__":
    run_benchmark()
