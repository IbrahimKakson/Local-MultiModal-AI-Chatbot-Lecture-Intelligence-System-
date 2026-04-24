import pytest
from unittest.mock import patch
from app.services.rag_chain import run_rag


@patch("app.services.rag_chain.generate_answer_from_model")
def test_run_rag_returns_string(mock_generate):
    mock_generate.return_value = "This is a mock answer."
    res = run_rag("What is AI?", top_k=3)
    assert isinstance(res, str)
    assert res == "This is a mock answer."
