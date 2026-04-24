import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

@patch("app.api.endpoints.PDFService")
def test_upload_pdf(MockPDFService):
    # Mock the return value to avoid running actual parsing
    mock_instance = MockPDFService.return_value
    c1 = MagicMock()
    c1.text = "chunk1"
    c2 = MagicMock()
    c2.text = "chunk2"
    mock_instance.extract_text_from_pdf.return_value = [c1, c2]
    
    dummy_pdf_content = b"%PDF-1.4 dummy pdf content"
    
    print(f"\n\n[API COMMUNICATION TEST - Phase 2, Step 4]")
    print("Testing FastAPI Upload Endpoint for PDF...")
    
    response = client.post(
        "/api/upload",
        files={"file": ("dummy.pdf", dummy_pdf_content, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"   -> Endpoint returned Status Code: {response.status_code}")
    print(f"   -> Uploaded File: {data['filename']}")
    print(f"   -> Processing Type: {data['type'].upper()}")
    print(f"   -> Result: Successfully triggered background parsing and extracted {data['chunks_extracted']} chunks!")
    
    assert data["filename"] == "dummy.pdf"
    assert data["type"] == "pdf"
    assert data["chunks_extracted"] == 2

@patch("app.api.endpoints.AudioService")
def test_upload_audio(MockAudioService, tmp_path):
    mock_instance = MockAudioService.return_value
    mock_instance.transcribe_audio.return_value = [{"text": "hello", "start": 0.0, "end": 1.0}]
    
    dummy_audio_content = b"fake audio bytes"
    
    print("\nTesting FastAPI Upload Endpoint for Audio...")
    
    response = client.post(
        "/api/upload",
        files={"file": ("dummy.mp3", dummy_audio_content, "audio/mpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"   -> Endpoint returned Status Code: {response.status_code}")
    print(f"   -> Uploaded File: {data['filename']}")
    print(f"   -> Processing Type: {data['type'].upper()}")
    print(f"   -> Result: Successfully triggered background whisper transcription and extracted {data['segments_extracted']} segments!")
    
    assert data["filename"] == "dummy.mp3"
    assert data["type"] == "audio"
    assert data["segments_extracted"] == 1
