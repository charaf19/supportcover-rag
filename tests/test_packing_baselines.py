from __future__ import annotations

from collections.abc import Callable

from supportcover_rag.packing import pack_greedy_query_cover, pack_mmr
from supportcover_rag.types import PackedEvidence, SentenceCandidate


def _candidate(
    title: str,
    sentence_id: int,
    sentence_terms: set[str],
    *,
    question_terms: set[str] | None = None,
    token_count: int = 1,
    query_overlap: float = 0.0,
    paragraph_score: float = 0.0,
    title_overlap: float = 0.0,
) -> SentenceCandidate:
    return SentenceCandidate(
        title=title,
        sentence_id=sentence_id,
        text=" ".join(sorted(sentence_terms)),
        paragraph_rank=sentence_id,
        paragraph_score=paragraph_score,
        token_count=token_count,
        question_terms=question_terms or set(),
        title_terms=set(),
        sentence_terms=sentence_terms,
        raw_features={
            "query_overlap": query_overlap,
            "paragraph_score_norm": paragraph_score,
            "title_overlap": title_overlap,
        },
    )


def _assert_budget_determinism_and_unique_keys(
    packer: Callable[[], PackedEvidence],
    token_budget: int,
) -> None:
    first = packer()
    second = packer()

    assert first.support_keys == second.support_keys
    assert first.used_tokens <= token_budget
    assert len(first.support_keys) == len(set(first.support_keys))


def test_sentence_baselines_enforce_budget_and_are_deterministic() -> None:
    question_terms = {"alpha", "beta", "gamma"}
    candidates = [
        _candidate("A", 0, {"alpha"}, question_terms=question_terms, query_overlap=1.0),
        _candidate("A", 0, {"alpha"}, question_terms=question_terms, query_overlap=1.0),
        _candidate("B", 0, {"beta"}, question_terms=question_terms, query_overlap=0.8),
        _candidate("C", 0, {"gamma"}, question_terms=question_terms, token_count=2, query_overlap=0.7),
    ]

    _assert_budget_determinism_and_unique_keys(
        lambda: pack_mmr(candidates, token_budget=2, lambda_relevance=0.5),
        token_budget=2,
    )
    _assert_budget_determinism_and_unique_keys(
        lambda: pack_greedy_query_cover(candidates, token_budget=2),
        token_budget=2,
    )


def test_mmr_prefers_similarly_relevant_less_redundant_evidence() -> None:
    candidates = [
        _candidate("A", 0, {"anchor", "shared"}, query_overlap=1.0),
        _candidate("B", 0, {"anchor", "shared"}, query_overlap=0.96),
        _candidate("C", 0, {"independent"}, query_overlap=0.90),
    ]

    packed = pack_mmr(candidates, token_budget=2, lambda_relevance=0.5)

    assert packed.support_keys == [("A", 0), ("C", 0)]


def test_greedy_query_cover_prefers_new_question_terms() -> None:
    question_terms = {"alpha", "beta"}
    candidates = [
        _candidate("A", 0, {"alpha"}, question_terms=question_terms),
        _candidate("B", 0, {"alpha", "repeated"}, question_terms=question_terms),
        _candidate("C", 0, {"beta"}, question_terms=question_terms),
    ]

    packed = pack_greedy_query_cover(candidates, token_budget=2)

    assert packed.support_keys == [("A", 0), ("C", 0)]


def test_greedy_query_cover_ignores_relevance_title_and_redundancy_bonuses() -> None:
    question_terms = {"alpha", "beta"}
    candidates = [
        _candidate("Used", 0, {"alpha", "shared", "noise"}, question_terms=question_terms),
        _candidate("Used", 1, {"beta", "shared", "noise"}, question_terms=question_terms),
        _candidate(
            "Fresh",
            2,
            {"beta", "independent"},
            question_terms=question_terms,
            query_overlap=1.0,
            paragraph_score=1.0,
            title_overlap=1.0,
        ),
    ]

    packed = pack_greedy_query_cover(candidates, token_budget=2)

    assert packed.support_keys == [("Used", 0), ("Used", 1)]
