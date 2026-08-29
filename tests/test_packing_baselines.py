from __future__ import annotations

from collections.abc import Callable

import pytest

from supportcover_rag.config import AppConfig, GenerationConfig, SupportCoverConfig
from supportcover_rag.packing import SupportCoverSelector, pack_greedy_query_cover, pack_mmr, pack_relevance_only
from supportcover_rag.pipeline import SUPPORTED_METHODS, ExperimentRunner
from supportcover_rag.types import HotpotExample, PackedEvidence, Paragraph, SelectedSentence, SentenceCandidate


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
    _assert_budget_determinism_and_unique_keys(
        lambda: pack_relevance_only(candidates, token_budget=2),
        token_budget=2,
    )
    _assert_budget_determinism_and_unique_keys(
        lambda: SupportCoverSelector(SupportCoverConfig()).select(candidates, token_budget=2),
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


def test_mmr_uses_the_same_base_relevance_as_supportcover() -> None:
    candidates = [
        _candidate("A", 0, {"alpha"}, query_overlap=0.2, paragraph_score=0.1, title_overlap=0.0),
        _candidate("B", 0, {"beta"}, query_overlap=0.1, paragraph_score=1.0, title_overlap=1.0),
    ]
    supportcover = SupportCoverSelector(
        SupportCoverConfig(
            beta_coverage=0.0,
            title_bonus=0.0,
            gamma_redundancy=0.0,
            delta_token_cost=0.0,
        )
    ).select(candidates, token_budget=1)

    mmr = pack_mmr(candidates, token_budget=1, lambda_relevance=1.0)

    assert mmr.support_keys == supportcover.support_keys == [("B", 0)]


def test_greedy_query_cover_prefers_new_question_terms() -> None:
    question_terms = {"alpha", "beta"}
    candidates = [
        _candidate("A", 0, {"alpha"}, question_terms=question_terms),
        _candidate("B", 0, {"alpha", "repeated"}, question_terms=question_terms),
        _candidate("C", 0, {"beta"}, question_terms=question_terms),
    ]

    packed = pack_greedy_query_cover(candidates, token_budget=2)

    assert packed.support_keys == [("A", 0), ("C", 0)]


def test_greedy_query_cover_uses_new_terms_per_token() -> None:
    question_terms = {"alpha", "beta"}
    candidates = [
        _candidate("Long", 0, {"alpha", "beta"}, question_terms=question_terms, token_count=3),
        _candidate("Short", 0, {"alpha"}, question_terms=question_terms, token_count=1),
    ]

    packed = pack_greedy_query_cover(candidates, token_budget=3)

    assert packed.support_keys == [("Short", 0)]


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


def _example() -> HotpotExample:
    return HotpotExample(
        example_id="example-1",
        question="Who wrote Hamlet?",
        answer="William Shakespeare",
        qtype="bridge",
        level="easy",
        context=[Paragraph(title="Hamlet", sentences=["Hamlet was written by William Shakespeare."])],
        supporting_facts=[("Hamlet", 0)],
    )


class _RecordingCompressor:
    def __init__(self, result: object) -> None:
        self.result = result
        self.call: dict[str, object] | None = None

    def compress(self, *, question: str, retrieved_paragraphs: object, token_budget: int) -> object:
        self.call = {
            "question": question,
            "retrieved_paragraphs": retrieved_paragraphs,
            "token_budget": token_budget,
        }
        return self.result


def test_external_compressor_receives_fair_inputs_and_returns_packed_evidence() -> None:
    packed = PackedEvidence(method="external_compressor", selected=[], token_budget=7)
    compressor = _RecordingCompressor(packed)
    runner = ExperimentRunner(
        AppConfig(generation=GenerationConfig(backend="echo")),
        external_compressor=compressor,  # type: ignore[arg-type]
    )

    result, _, _, metadata = runner._build_packed_evidence(
        _example(),
        method="external_compressor",
        token_budget=7,
        retrieval_depth=1,
    )

    assert result is packed
    assert compressor.call is not None
    assert compressor.call["question"] == _example().question
    assert compressor.call["token_budget"] == 7
    assert len(compressor.call["retrieved_paragraphs"]) == 1  # type: ignore[arg-type]
    assert metadata["num_candidates"] == 0


def test_external_compressor_requires_an_injected_adapter() -> None:
    runner = ExperimentRunner(AppConfig(generation=GenerationConfig(backend="echo")))

    with pytest.raises(RuntimeError, match="requires an injected"):
        runner._build_packed_evidence(
            _example(),
            method="external_compressor",
            token_budget=7,
            retrieval_depth=1,
        )


def test_external_compressor_rejects_wrong_return_type() -> None:
    runner = ExperimentRunner(
        AppConfig(generation=GenerationConfig(backend="echo")),
        external_compressor=_RecordingCompressor(object()),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="must return PackedEvidence"):
        runner._build_packed_evidence(
            _example(),
            method="external_compressor",
            token_budget=7,
            retrieval_depth=1,
        )


def test_external_compressor_rejects_token_budget_violation() -> None:
    over_budget_candidate = _candidate("External", 0, {"evidence"}, token_count=8)
    packed = PackedEvidence(
        method="external_compressor",
        selected=[
            SelectedSentence(
                candidate=over_budget_candidate,
                score=1.0,
                contributions={},
            )
        ],
        token_budget=7,
    )
    runner = ExperimentRunner(
        AppConfig(generation=GenerationConfig(backend="echo")),
        external_compressor=_RecordingCompressor(packed),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exceeded the token budget"):
        runner._build_packed_evidence(
            _example(),
            method="external_compressor",
            token_budget=7,
            retrieval_depth=1,
        )


def test_publication_grade_methods_are_registered() -> None:
    assert {
        "paragraph_topk",
        "relevance_only",
        "mmr_sentence",
        "greedy_query_cover",
        "external_compressor",
        "supportcover",
        "supportcover_final",
    } <= set(SUPPORTED_METHODS)
