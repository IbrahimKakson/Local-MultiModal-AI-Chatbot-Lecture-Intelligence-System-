import os
from faster_whisper import WhisperModel

# Use a global model instance or load it explicitly.
# For local dev, "tiny" or "base" is recommended, running on "cpu"
# If a GPU is available, it will try to use "cuda" or "auto" with float16 support.
# To keep CPU performance acceptable without GPU, use 'int8'.

MODEL_SIZE = "tiny" 

class AudioService:
    def __init__(self, model_size=MODEL_SIZE):
        """Initialize the Audio service."""
        pass


    def transcribe_audio(self, file_path: str) -> list[dict]:
        """
        Transcribe an audio file and return text segments with timestamps.
        Expected output format: [{"text": "...", "start": 0.0, "end": 5.0}]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load model only when needed to save RAM
        print("Loading Whisper model into RAM for transcription...")
        import multiprocessing
        # Restrict Whisper to use at most 75% of CPU cores (leaving at least 1-2 for the webserver)
        # This prevents the Chainlit websocket from dropping heartbeat packets and crashing the browser tab!
        optimal_threads = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=optimal_threads)

        # Reverted back to deep mathematical search (beam_size=5) for maximum accuracy.
        # This takes 5x longer but completely stops 'stupid' hallucinated outputs.
        segments, info = model.transcribe(file_path, beam_size=5)
        
        result = []
        # Convert generator to a list so we can slice it
        segments_list = list(segments)
        
        # Group 8 tiny audio segments together into 1 cohesive "paragraph" block
        # We overlap by 2 segments to ensure sentences aren't cleanly cut in half across blocks
        CHUNK_SIZE = 8
        OVERLAP_SIZE = 2
        
        i = 0
        while i < len(segments_list):
            chunk_segs = segments_list[i:i + CHUNK_SIZE]
            if not chunk_segs:
                break
                
            combined_text = " ".join([seg.text.strip() for seg in chunk_segs if seg.text.strip()])
            
            if combined_text:
                result.append({
                    "text": combined_text,
                    "start": chunk_segs[0].start,
                    "end": chunk_segs[-1].end
                })
                
            # Advance the sliding window
            i += (CHUNK_SIZE - OVERLAP_SIZE)

            
        # Free memory immediately
        print("Transcription complete. Unloading Whisper from RAM...")
        del model
        import gc
        gc.collect()

        return result
