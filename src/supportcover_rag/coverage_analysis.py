from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from statistics import mean
from typing import Any

import numpy as np

from supportcover_rag.statistics import bootstrap_ci, percentile_confidence_interval
from supportcover_rag.text import informative_term_set, jaccard_similarity
from supportcover_rag.types import HotpotExample, PackedEvidence, SupportKey


DiagnosticRecord = Mapping[str, Any]
DiagnosticExtractor = Callable[[DiagnosticRecord], float]


def question_term_coverage(question: str, packed: PackedEvidence) -> float:
    question_terms = informative_term_set(question)
    if not question_terms:
        return 1.0
    selected_terms = set().union(*(item.candidate.sentence_terms for item in packed.selected)) if packed.selected else set()
    return len(question_terms & selected_terms) / len(question_terms)


def unique_title_count(packed: PackedEvidence) -> int:
    return len({item.candidate.title for item in packed.selected})


def unique_title_ratio(packed: PackedEvidence) -> float:
    if not packed.selected:
        return 0.0
    return unique_title_count(packed) / len(packed.selected)


def selected_sentence_redundancy(packed: PackedEvidence) -> tuple[float, float]:
    similarities = [
        jaccard_similarity(left.candidate.sentence_terms, right.candidate.sentence_terms)
        for left, right in combinations(packed.selected, 2)
    ]
    if not similarities:
        return 0.0, 0.0
    return mean(similarities), max(similarities)


def gold_support_recall(selected_support: Sequence[SupportKey], gold_support: Sequence[SupportKey]) -> float:
    gold = set(gold_support)
    if not gold:
        return 1.0
    return len(set(selected_support) & gold) / len(gold)


def build_coverage_diagnostics(example: HotpotExample, packed: PackedEvidence) -> dict[str, float | int | str]:
    mean_redundancy, max_redundancy = selected_sentence_redundancy(packed)
    return {
        "example_id": example.example_id,
        "question_term_coverage": question_term_coverage(example.question, packed),
        "unique_title_count": unique_title_count(packed),
        "unique_title_ratio": unique_title_ratio(packed),
        "mean_selected_sentence_redundancy": mean_redundancy,
        "max_selected_sentence_redundancy": max_redundancy,
        "gold_support_recall": gold_support_recall(packed.support_keys, example.supporting_facts),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths.")
    if len(left) < 2:
        raise ValueError("Spearman correlation requires at least two paired values.")
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("Spearman inputs must be finite.")
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    if float(np.std(left_ranks)) == 0.0 or float(np.std(right_ranks)) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def bootstrap_diagnostic_mean_ci(
    records: Sequence[DiagnosticRecord],
    extractor: DiagnosticExtractor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    return bootstrap_ci(
        records,
        extractor,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


def spearman_with_bootstrap_ci(
    records: Sequence[DiagnosticRecord],
    left_extractor: DiagnosticExtractor,
    right_extractor: DiagnosticExtractor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    left = np.asarray([float(left_extractor(record)) for record in records], dtype=float)
    right = np.asarray([float(right_extractor(record)) for record in records], dtype=float)
    correlation = spearman_correlation(left, right)
    rng = np.random.default_rng(seed)
    bootstrap_correlations = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_indices = rng.integers(0, len(records), size=len(records))
        bootstrap_correlations[index] = spearman_correlation(left[sampled_indices], right[sampled_indices])
    ci_low, ci_high = percentile_confidence_interval(bootstrap_correlations, confidence)
    return {
        "spearman": correlation,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(records),
    }
