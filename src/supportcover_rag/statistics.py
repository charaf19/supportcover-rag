from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


PredictionRecordLike = Mapping[str, Any]
MetricExtractor = Callable[[PredictionRecordLike], float]


def _index_records(
    records: Sequence[PredictionRecordLike],
    *,
    collection_name: str,
) -> tuple[list[str], dict[str, PredictionRecordLike]]:
    ordered_ids: list[str] = []
    indexed: dict[str, PredictionRecordLike] = {}
    for index, record in enumerate(records):
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{collection_name} record {index} is missing a non-empty string example_id.")
        if example_id in indexed:
            raise ValueError(f"Duplicate example ID in {collection_name}: {example_id}")
        ordered_ids.append(example_id)
        indexed[example_id] = record
    return ordered_ids, indexed


def align_records_by_example_id(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
) -> list[tuple[PredictionRecordLike, PredictionRecordLike]]:
    """Align two collections in the deterministic order of the first collection."""
    ids_a, indexed_a = _index_records(records_a, collection_name="method A")
    ids_b, indexed_b = _index_records(records_b, collection_name="method B")
    missing_from_b = [example_id for example_id in ids_a if example_id not in indexed_b]
    missing_from_a = [example_id for example_id in ids_b if example_id not in indexed_a]
    if missing_from_a or missing_from_b:
        details: list[str] = []
        if missing_from_b:
            details.append(f"missing from method B: {', '.join(missing_from_b)}")
        if missing_from_a:
            details.append(f"missing from method A: {', '.join(missing_from_a)}")
        raise ValueError("Prediction populations do not match: " + "; ".join(details))
    return [(indexed_a[example_id], indexed_b[example_id]) for example_id in ids_a]


def _metric_values(records: Sequence[PredictionRecordLike], extractor: MetricExtractor) -> np.ndarray:
    if not records:
        raise ValueError("At least one prediction record is required.")
    values = np.asarray([float(extractor(record)) for record in records], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Metric values must be finite.")
    return values


def _confidence_fraction(confidence: float) -> float:
    value = confidence / 100.0 if confidence > 1.0 else confidence
    if not 0.0 < value < 1.0:
        raise ValueError("confidence must be between 0 and 1, or between 0 and 100 percent.")
    return value


def percentile_confidence_interval(
    samples: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    if not len(values):
        raise ValueError("At least one bootstrap sample is required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap samples must be finite.")
    confidence_fraction = _confidence_fraction(confidence)
    tail = (1.0 - confidence_fraction) * 50.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_ci(
    records: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    _confidence_fraction(confidence)
    values = _metric_values(records, metric_extractor)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_indices = rng.integers(0, len(values), size=len(values))
        sampled_means[index] = float(np.mean(values[sampled_indices]))
    return percentile_confidence_interval(sampled_means, confidence)


def paired_bootstrap(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    _confidence_fraction(confidence)
    aligned = align_records_by_example_id(records_a, records_b)
    differences = np.asarray(
        [float(metric_extractor(record_a)) - float(metric_extractor(record_b)) for record_a, record_b in aligned],
        dtype=float,
    )
    if not len(differences):
        raise ValueError("At least one paired prediction is required.")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired metric differences must be finite.")

    rng = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_indices = rng.integers(0, len(differences), size=len(differences))
        sampled_means[index] = float(np.mean(differences[sampled_indices]))
    return percentile_confidence_interval(sampled_means, confidence)


def paired_random_sign_test(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    *,
    permutations: int = 10_000,
    seed: int = 42,
) -> float:
    if permutations <= 0:
        raise ValueError("permutations must be positive.")
    aligned = align_records_by_example_id(records_a, records_b)
    differences = np.asarray(
        [float(metric_extractor(record_a)) - float(metric_extractor(record_b)) for record_a, record_b in aligned],
        dtype=float,
    )
    if not len(differences):
        raise ValueError("At least one paired prediction is required.")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired metric differences must be finite.")

    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        if abs(float(np.mean(differences * signs))) >= observed:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (permutations + 1)


def paired_standardized_effect_size(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
) -> float:
    aligned = align_records_by_example_id(records_a, records_b)
    differences = np.asarray(
        [float(metric_extractor(record_a)) - float(metric_extractor(record_b)) for record_a, record_b in aligned],
        dtype=float,
    )
    if not len(differences):
        raise ValueError("At least one paired prediction is required.")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired metric differences must be finite.")

    mean_difference = float(np.mean(differences))
    if len(differences) < 2:
        return 0.0 if mean_difference == 0.0 else math.copysign(math.inf, mean_difference)
    standard_deviation = float(np.std(differences, ddof=1))
    if standard_deviation == 0.0:
        return 0.0 if mean_difference == 0.0 else math.copysign(math.inf, mean_difference)
    return mean_difference / standard_deviation


def build_comparison_row(
    *,
    metric: str,
    method_a: str,
    method_b: str,
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    resamples: int = 10_000,
    confidence: float = 0.95,
    permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    aligned = align_records_by_example_id(records_a, records_b)
    values_a = np.asarray([float(metric_extractor(record_a)) for record_a, _ in aligned], dtype=float)
    values_b = np.asarray([float(metric_extractor(record_b)) for _, record_b in aligned], dtype=float)
    if not len(aligned):
        raise ValueError("At least one paired prediction is required.")
    if not np.all(np.isfinite(values_a)) or not np.all(np.isfinite(values_b)):
        raise ValueError("Metric values must be finite.")

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    ci_low, ci_high = paired_bootstrap(
        records_a,
        records_b,
        metric_extractor,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_a - mean_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": paired_random_sign_test(
            records_a,
            records_b,
            metric_extractor,
            permutations=permutations,
            seed=seed,
        ),
        "effect_size": paired_standardized_effect_size(records_a, records_b, metric_extractor),
        "n": len(aligned),
    }
