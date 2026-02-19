import pytest
from app.services.pdf_service import PDFService
from unittest.mock import MagicMock, patch
from pypdf import PdfWriter
import io

@pytest.fixture
def sample_pdf_bytes():
    # Create a simple PDF in memory for testing
    buffer = io.BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    # Adding annotation or text is hard programmatically with pypdf without reading source
    # So we will mock the PdfReader instead for simpler unit testing logic.
    writer.write(buffer)
    return buffer.getvalue()

def test_extract_text_empty(sample_pdf_bytes):
    service = PDFService()
    # Mocking PdfReader to simulate text extraction behavior
    with patch("app.services.pdf_service.PdfReader") as MockReader:
        mock_instance = MockReader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_instance.pages = [mock_page]
        
        chunks = service.extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        assert len(chunks) == 0

def test_extract_text_success(sample_pdf_bytes):
    service = PDFService()
    
    # Mocking a PDF with some text
    with patch("app.services.pdf_service.PdfReader") as MockReader:
        mock_instance = MockReader.return_value
        mock_page = MagicMock()
        # Create enough text to potentially cause a split if chunk size was small, 
        # but with default 1000 it might be one chunk.
        mock_page.extract_text.return_value = "Hello World. " * 50
        mock_instance.pages = [mock_page]

        chunks = service.extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        assert len(chunks) > 0
        assert chunks[0].source == "test.pdf"
        assert "Hello World" in chunks[0].text
