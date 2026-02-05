from app.services.rag_chain import run_rag


def test_run_rag_returns_string():
    res = run_rag("What is AI?", top_k=3)
    assert isinstance(res, str)
