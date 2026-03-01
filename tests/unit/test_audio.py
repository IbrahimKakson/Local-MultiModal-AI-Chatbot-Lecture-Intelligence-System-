import pytest
from app.services.audio_service import AudioService
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def mock_whisper_model():
    """Mock the WhisperModel to avoid downloading actual models during testing."""
    with patch("app.services.audio_service.WhisperModel") as MockModel:
        # Create a mock instance
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield mock_instance

def test_transcribe_audio_file_not_found(mock_whisper_model):
    service = AudioService(model_size="tiny")
    with pytest.raises(FileNotFoundError):
        service.transcribe_audio("non_existent_file.mp3")

def test_transcribe_audio_success(mock_whisper_model, tmp_path):
    # Setup dummy segments to return from the mock
    mock_segment_1 = MagicMock()
    mock_segment_1.text = " Hello world."
    mock_segment_1.start = 0.0
    mock_segment_1.end = 2.5
    
    mock_segment_2 = MagicMock()
    mock_segment_2.text = " This is a test."
    mock_segment_2.start = 2.5
    mock_segment_2.end = 4.0

    mock_whisper_model.transcribe.return_value = ([mock_segment_1, mock_segment_2], {})
    
    service = AudioService(model_size="tiny")
    
    # Create a dummy valid file using pytest's tmp_path feature
    dummy_audio = tmp_path / "dummy.mp3"
    dummy_audio.write_text("dummy mp3 content")
    
    # Run the transcription methods
    results = service.transcribe_audio(str(dummy_audio))
    
    assert len(results) == 2
    assert results[0]["text"] == "Hello world."
    assert results[0]["start"] == 0.0
    assert "start" in results[1]
    assert "end" in results[1]
