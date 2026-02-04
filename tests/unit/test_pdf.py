from app.services.pdf_service import extract_text_from_pdf


def test_extract_pdf():
    assert isinstance(extract_text_from_pdf("dummy.pdf"), str)
