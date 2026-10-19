from pathlib import Path

from inspectflow.rag.assistant import answer_question, load_corpus


def test_sop_answer_cites_source() -> None:
    corpus = load_corpus(Path(__file__).resolve().parents[1] / "docs" / "corpus")
    answer = answer_question("Khi nào lô bị reject vì xước?", corpus)
    assert "scratch" in answer.answer.lower() or "reject" in answer.answer.lower()
    assert answer.citations
