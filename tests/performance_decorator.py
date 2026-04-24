import time
from functools import wraps

def measure_generation_performance(func):
    """
    Decorator to measure Time-To-First-Token (TTFT) and Total Generation Time (TGT)
    for generator functions (like streaming LLM outputs).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        first_token_time = None
        
        generator = func(*args, **kwargs)
        
        try:
            for item in generator:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    print(f"\n[Performance] Time-To-First-Token (TTFT): {ttft:.4f} seconds")
                yield item
        finally:
            end_time = time.time()
            tgt = end_time - start_time
            print(f"\n[Performance] Total Generation Time (TGT): {tgt:.4f} seconds")
            if first_token_time is not None:
                print(f"[Performance] Tokens generated per second: ~ {len(args) / tgt if len(args) else 'N/A'}")

    return wrapper
