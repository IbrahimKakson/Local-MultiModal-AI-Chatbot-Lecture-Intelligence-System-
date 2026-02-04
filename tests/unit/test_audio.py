from app.services.audio_service import transcribe_audio


def test_transcribe_audio():
    assert isinstance(transcribe_audio("dummy.mp3"), str)
