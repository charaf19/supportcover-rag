from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SupportKey = tuple[str, int]


@dataclass(slots=True)
class Paragraph:
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences if s.strip())


@dataclass(slots=True)
class HotpotExample:
    example_id: str
    question: str
    answer: str
    qtype: str
    level: str
    context: list[Paragraph]
    supporting_facts: list[SupportKey]


@dataclass(slots=True)
class RetrievedParagraph:
    title: str
    sentences: list[str]
    text: str
    rank: int
    score: float


@dataclass(slots=True)
class SentenceCandidate:
    title: str
    sentence_id: int
    text: str
    paragraph_rank: int
    paragraph_score: float
    token_count: int
    question_terms: set[str]
    title_terms: set[str]
    sentence_terms: set[str]
    raw_features: dict[str, float] = field(default_factory=dict)

    @property
    def support_key(self) -> SupportKey:
        return (self.title, self.sentence_id)


@dataclass(slots=True)
class SelectedSentence:
    candidate: SentenceCandidate
    score: float
    contributions: dict[str, float]


@dataclass(slots=True)
class PackedEvidence:
    method: str
    selected: list[SelectedSentence]
    token_budget: int

    @property
    def used_tokens(self) -> int:
        return sum(item.candidate.token_count for item in self.selected)

    @property
    def support_keys(self) -> list[SupportKey]:
        return [item.candidate.support_key for item in self.selected]

    def render(self, include_titles: bool = True) -> str:
        if not self.selected:
            return ""
        grouped: dict[str, list[str]] = {}
        for item in self.selected:
            grouped.setdefault(item.candidate.title, []).append(item.candidate.text)
        blocks: list[str] = []
        for title, sentences in grouped.items():
            if include_titles:
                blocks.append(f"Title: {title}\n- " + "\n- ".join(sentences))
            else:
                blocks.append("- " + "\n- ".join(sentences))
        return "\n\n".join(blocks)


@dataclass(slots=True)
class PredictionRecord:
    example_id: str
    method: str
    token_budget: int
    question: str
    gold_answer: str
    predicted_answer: str
    gold_supporting_facts: list[SupportKey]
    predicted_supporting_facts: list[SupportKey]
    answer_em: float
    answer_f1: float
    support_em: float
    support_precision: float
    support_recall: float
    support_f1: float
    coverage_at_budget: float
    evidence_tokens: int
    retrieval_latency_ms: float
    packing_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    peak_rss_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "method": self.method,
            "token_budget": self.token_budget,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "gold_supporting_facts": list(self.gold_supporting_facts),
            "predicted_supporting_facts": list(self.predicted_supporting_facts),
            "answer_em": self.answer_em,
            "answer_f1": self.answer_f1,
            "support_em": self.support_em,
            "support_precision": self.support_precision,
            "support_recall": self.support_recall,
            "support_f1": self.support_f1,
            "coverage_at_budget": self.coverage_at_budget,
            "evidence_tokens": self.evidence_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "packing_latency_ms": self.packing_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "peak_rss_mb": self.peak_rss_mb,
            "metadata": self.metadata,
        }
