from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from supportcover_rag.statistics import (
    align_records_by_example_id,
    bootstrap_ci,
    build_comparison_row,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap,
    paired_random_sign_test,
    paired_standardized_effect_size,
)


def _records(values: list[float]) -> list[dict[str, float | str]]:
    return [{"example_id": f"id-{index}", "score": value} for index, value in enumerate(values)]


def _score(record: Mapping[str, object]) -> float:
    return float(record["score"])


def test_identical_methods_have_zero_paired_delta() -> None:
    records = _records([0.0, 0.5, 1.0, 0.5])

    row = build_comparison_row(
        metric="score",
        method_a="a",
        method_b="b",
        records_a=records,
        records_b=list(reversed(records)),
        metric_extractor=_score,
        resamples=100,
        permutations=100,
        seed=7,
    )

    assert row["delta"] == 0.0
    assert row["ci_low"] == 0.0
    assert row["ci_high"] == 0.0
    assert row["p_value"] == 1.0
    assert row["effect_size"] == 0.0


def test_consistent_positive_paired_effect() -> None:
    records_a = _records([1.0, 2.0, 3.0, 4.0])
    records_b = _records([0.0, 1.0, 2.0, 3.0])

    ci_low, ci_high = paired_bootstrap(records_a, records_b, _score, resamples=100, seed=3)

    assert ci_low == 1.0
    assert ci_high == 1.0
    assert paired_standardized_effect_size(records_a, records_b, _score) == math.inf


def test_alignment_rejects_duplicate_ids() -> None:
    duplicate = [{"example_id": "same", "score": 1.0}, {"example_id": "same", "score": 2.0}]

    with pytest.raises(ValueError, match="Duplicate"):
        align_records_by_example_id(duplicate, _records([1.0, 2.0]))


def test_alignment_reports_mismatched_ids() -> None:
    records_a = [{"example_id": "a", "score": 1.0}]
    records_b = [{"example_id": "b", "score": 1.0}]

    with pytest.raises(ValueError, match="missing from method"):
        align_records_by_example_id(records_a, records_b)


def test_seeded_bootstrap_and_permutation_are_deterministic() -> None:
    records_a = _records([0.2, 0.4, 0.8, 1.0])
    records_b = _records([0.1, 0.5, 0.7, 0.9])

    assert bootstrap_ci(records_a, _score, resamples=100, seed=11) == bootstrap_ci(
        records_a,
        _score,
        resamples=100,
        seed=11,
    )
    assert paired_random_sign_test(
        records_a,
        records_b,
        _score,
        permutations=100,
        seed=11,
    ) == paired_random_sign_test(records_a, records_b, _score, permutations=100, seed=11)


def test_zero_variance_zero_effect_is_safe() -> None:
    records = _records([1.0, 1.0, 1.0])

    assert paired_standardized_effect_size(records, records, _score) == 0.0


def test_exact_mcnemar_reports_known_discordant_counts() -> None:
    records_a = _records([1.0, 1.0, 1.0, 0.0, 0.0])
    records_b = _records([0.0, 0.0, 1.0, 1.0, 0.0])

    result = mcnemar_exact(records_a, records_b, _score)

    assert result["a_correct_b_wrong"] == 2
    assert result["a_wrong_b_correct"] == 1
    assert result["discordant"] == 3
    assert result["p_value"] == 1.0


def test_exact_mcnemar_detects_one_sided_discordance() -> None:
    result = mcnemar_exact(_records([1.0] * 10), _records([0.0] * 10), _score)

    assert result["a_correct_b_wrong"] == 10
    assert result["a_wrong_b_correct"] == 0
    assert result["p_value"] == pytest.approx(2 / 1024)


def test_holm_bonferroni_preserves_order_and_monotonicity() -> None:
    adjusted = holm_bonferroni([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_exact_random_sign_test_detects_strong_paired_improvement() -> None:
    records_a = _records([1.0] * 12)
    records_b = _records([0.0] * 12)

    p_value = paired_random_sign_test(records_a, records_b, _score, permutations=100, seed=17)

    assert p_value < 0.01
