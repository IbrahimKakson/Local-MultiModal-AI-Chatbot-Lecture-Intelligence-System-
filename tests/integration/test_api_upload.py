import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)

@patch("app.api.endpoints.PDFService")
def test_upload_pdf(MockPDFService):
    # Mock the return value to avoid running actual parsing
    mock_instance = MockPDFService.return_value
    mock_instance.extract_text_from_pdf.return_value = ["chunk1", "chunk2"]
    
    dummy_pdf_content = b"%PDF-1.4 dummy pdf content"
    
    response = client.post(
        "/api/upload",
        files={"file": ("dummy.pdf", dummy_pdf_content, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "dummy.pdf"
    assert data["type"] == "pdf"
    assert data["chunks_extracted"] == 2

@patch("app.api.endpoints.AudioService")
def test_upload_audio(MockAudioService, tmp_path):
    mock_instance = MockAudioService.return_value
    mock_instance.transcribe_audio.return_value = [{"text": "hello", "start": 0.0, "end": 1.0}]
    
    dummy_audio_content = b"fake audio bytes"
    
    response = client.post(
        "/api/upload",
        files={"file": ("dummy.mp3", dummy_audio_content, "audio/mpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "dummy.mp3"
    assert data["type"] == "audio"
    assert data["segments_extracted"] == 1
