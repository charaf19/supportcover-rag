from __future__ import annotations

from supportcover_rag.packing import pack_paragraphs
from supportcover_rag.types import PackedEvidence, RetrievedParagraph, SelectedSentence, SentenceCandidate


class WhitespaceTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _selected_sentence() -> SelectedSentence:
    candidate = SentenceCandidate(
        title="Fallback",
        sentence_id=3,
        text="fallback evidence",
        paragraph_rank=0,
        paragraph_score=1.0,
        token_count=2,
        question_terms=set(),
        title_terms=set(),
        sentence_terms={"fallback", "evidence"},
    )
    return SelectedSentence(candidate=candidate, score=1.0, contributions={})


def test_packed_evidence_explicit_support_keys_override_derived_keys() -> None:
    selected = [_selected_sentence()]

    derived = PackedEvidence(method="derived", selected=selected, token_budget=4)
    explicit = PackedEvidence(
        method="explicit",
        selected=selected,
        token_budget=4,
        explicit_support_keys=[("Paragraph", 0), ("Paragraph", 1)],
    )

    assert derived.support_keys == [("Fallback", 3)]
    assert explicit.support_keys == [("Paragraph", 0), ("Paragraph", 1)]


def test_paragraph_packing_exposes_every_included_sentence_support_key() -> None:
    paragraphs = [
        RetrievedParagraph(
            title="First",
            sentences=["one", "two words"],
            text="one two words",
            rank=0,
            score=2.0,
        ),
        RetrievedParagraph(
            title="Second",
            sentences=["three", "four"],
            text="three four",
            rank=1,
            score=1.0,
        ),
    ]

    packed = pack_paragraphs(
        question="question",
        retrieved_paragraphs=paragraphs,
        token_budget=3,
        token_counter=WhitespaceTokenCounter(),
    )

    assert packed.support_keys == [("First", 0), ("First", 1)]
    assert len(packed.support_keys) == len(set(packed.support_keys))
    assert packed.used_tokens == 3
    assert "one two words" in packed.render()
