"""Extractive RAG over station SOPs.

Retrieval is lexical and runs offline. The generator only quotes a retrieved
passage. Point `generate` at an LLM later; keep the same citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from inspectflow.schemas import ChatAnswer

_TOKEN = re.compile(r"[a-z0-9à-ỹ]+", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    source: str
    heading: str
    text: str


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text)}


def load_corpus(directory: Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        heading = path.stem
        buffer: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if buffer:
                    passages.append(Passage(path.name, heading, "\n".join(buffer).strip()))
                heading = line.removeprefix("## ").strip()
                buffer = []
                continue
            buffer.append(line)
        if buffer and any(part.strip() for part in buffer):
            passages.append(Passage(path.name, heading, "\n".join(buffer).strip()))
    return passages


def _score(query: set[str], passage: Passage, document_frequency: dict[str, int]) -> float:
    overlap = query & _tokens(passage.heading + " " + passage.text)
    return sum(1.0 / document_frequency[token] for token in overlap)


def answer_question(question: str, passages: list[Passage], *, top_k: int = 3) -> ChatAnswer:
    query = _tokens(question)
    if not query or not passages:
        return ChatAnswer(answer="Chưa có đoạn SOP phù hợp với câu hỏi.", citations=[])
    document_frequency: dict[str, int] = {}
    passage_tokens = [_tokens(passage.heading + " " + passage.text) for passage in passages]
    for tokens in passage_tokens:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    ranked = sorted(
        passages,
        key=lambda passage: _score(query, passage, document_frequency),
        reverse=True,
    )
    chosen = [passage for passage in ranked[:top_k] if query & _tokens(passage.heading + " " + passage.text)]
    if not chosen:
        return ChatAnswer(answer="Không tìm thấy nội dung này trong SOP trạm kiểm tra.", citations=[])
    best = chosen[0]
    snippet = best.text.strip().splitlines()[0][:400]
    citations = [f"{passage.source}#{passage.heading}" for passage in chosen]
    return ChatAnswer(answer=f"{best.heading}: {snippet}", citations=citations)
