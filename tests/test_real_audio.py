from app.services.audio_service import AudioService
import os

def test_real_audio():
    # --- Instructions ---
    # 1. Place a real audio file (.mp3, .wav, .m4a) in this main project folder.
    # 2. Change the 'audio_path' variable below to match the name of your file.
    
    audio_path = "data/uploads/sample.mp4" 
    
    # --------------------

    if not os.path.exists(audio_path):
        print(f"[ERROR] Could not find '{audio_path}'.")
        print(f"Please place an audio file in the project folder and make sure the name matches '{audio_path}'.")
        return

    print(f"[START] Loading Whisper model and transcribing '{audio_path}'...")
    print("(Note: It might take a minute the very first time as it downloads the AI model to your computer)")
    
    try:
        service = AudioService()
        segments = service.transcribe_audio(audio_path)
        
        print(f"\n[SUCCESS] Extracted {len(segments)} segments!")
        print("--- Preview ---")
        
        # Print up to the first 5 segments
        for i, seg in enumerate(segments[:5]):
            print(f"  [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
            
        if len(segments) > 5:
            print("  ... (and more)")
            
    except Exception as e:
        print(f"\n[ERROR] An error occurred while processing the audio: {e}")

if __name__ == "__main__":
    test_real_audio()
