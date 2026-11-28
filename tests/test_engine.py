"""Basic smoke tests. Run with: pytest -q"""
from src.pipeline import get_engine
from src.preprocessing import clean_text, preprocess_query


def test_clean_text():
    assert clean_text("  Hello,   WORLD!! ") == "hello world"


def test_tokenize():
    assert preprocess_query("The Borrower shall pay.") == ["the", "borrower", "shall", "pay"]


def test_search_returns_sorted_scores():
    engine = get_engine()
    res = engine.search("The borrower shall not transfer its assets", top_n=5)
    assert len(res) == 5
    assert res["similarity_score"].is_monotonic_decreasing


def test_clause_type_filter():
    engine = get_engine()
    target = engine.clause_types[0]
    res = engine.search("payment within thirty days", top_n=5, clause_types=[target])
    assert set(res["clause_type"]) == {target}
