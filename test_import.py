import sys
import os

# Add the project root to sys.path so we can import 'app'
sys.path.append(os.getcwd())

try:
    from app.services.audio_service import AudioService
    service = AudioService(model_size="tiny")
    print("SUCCESS: AudioService initialized successfully!")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
