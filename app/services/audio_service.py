import os
from faster_whisper import WhisperModel

# Use a global model instance or load it explicitly.
# For local dev, "tiny" or "base" is recommended, running on "cpu"
# If a GPU is available, it will try to use "cuda" or "auto" with float16 support.
# To keep CPU performance acceptable without GPU, use 'int8'.

MODEL_SIZE = "tiny" 

class AudioService:
    def __init__(self, model_size=MODEL_SIZE):
        """Initialize the Whisper model."""
        # This will download the model on the first run to a local cache folder.
        
        # device="auto" tries to use GPU but falls back to CPU.
        # compute_type="int8" speeds up CPU inference.
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_audio(self, file_path: str) -> list[dict]:
        """
        Transcribe an audio file and return text segments with timestamps.
        Expected output format: [{"text": "...", "start": 0.0, "end": 5.0}]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        segments, info = self.model.transcribe(file_path, beam_size=5)
        
        result = []
        for segment in segments:
            # Note: We can grab segment.start and segment.end along with the text.
            result.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })
            
        return result
