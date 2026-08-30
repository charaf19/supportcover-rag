from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np


PredictionRecordLike = Mapping[str, Any]
MetricExtractor = Callable[[PredictionRecordLike], float]
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_PERMUTATION_REPLICATES = 10_000
DEFAULT_STATISTICS_SEED = 42
DEFAULT_CONFIDENCE = 0.95
_RESAMPLING_BATCH_SIZE = 256


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


def paired_metric_values(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite metric arrays after strict example-ID pairing."""
    aligned = align_records_by_example_id(records_a, records_b)
    if not aligned:
        raise ValueError("At least one paired prediction is required.")
    values_a = np.asarray([float(metric_extractor(record_a)) for record_a, _ in aligned], dtype=float)
    values_b = np.asarray([float(metric_extractor(record_b)) for _, record_b in aligned], dtype=float)
    if not np.all(np.isfinite(values_a)) or not np.all(np.isfinite(values_b)):
        raise ValueError("Metric values must be finite.")
    return values_a, values_b


def _resampled_means(
    values: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap scalar means in bounded vectorized batches."""
    sampled_means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, _RESAMPLING_BATCH_SIZE):
        stop = min(start + _RESAMPLING_BATCH_SIZE, resamples)
        sampled_indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled_means[start:stop] = np.mean(values[sampled_indices], axis=1)
    return sampled_means


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
    resamples: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_STATISTICS_SEED,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    _confidence_fraction(confidence)
    values = _metric_values(records, metric_extractor)
    rng = np.random.default_rng(seed)
    sampled_means = _resampled_means(values, resamples=resamples, rng=rng)
    return percentile_confidence_interval(sampled_means, confidence)


def paired_bootstrap(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_STATISTICS_SEED,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    _confidence_fraction(confidence)
    values_a, values_b = paired_metric_values(records_a, records_b, metric_extractor)
    differences = values_a - values_b

    rng = np.random.default_rng(seed)
    sampled_means = _resampled_means(differences, resamples=resamples, rng=rng)
    return percentile_confidence_interval(sampled_means, confidence)


def _exact_random_sign_p_value(differences: np.ndarray, observed: float) -> float:
    total = 1 << len(differences)
    at_least_as_extreme = 0
    bit_positions = np.arange(len(differences), dtype=np.uint64)
    for start in range(0, total, _RESAMPLING_BATCH_SIZE):
        stop = min(start + _RESAMPLING_BATCH_SIZE, total)
        combinations = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (((combinations >> bit_positions) & 1).astype(float) * 2.0) - 1.0
        permuted = np.abs(np.mean(signs * differences, axis=1))
        at_least_as_extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return at_least_as_extreme / total


def paired_random_sign_test(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
    *,
    permutations: int = DEFAULT_PERMUTATION_REPLICATES,
    seed: int = DEFAULT_STATISTICS_SEED,
    exact_threshold: int = 20,
) -> float:
    if permutations <= 0:
        raise ValueError("permutations must be positive.")
    values_a, values_b = paired_metric_values(records_a, records_b, metric_extractor)
    differences = values_a - values_b

    observed = abs(float(np.mean(differences)))
    if len(differences) <= exact_threshold:
        return _exact_random_sign_p_value(differences, observed)

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for start in range(0, permutations, _RESAMPLING_BATCH_SIZE):
        stop = min(start + _RESAMPLING_BATCH_SIZE, permutations)
        signs = rng.integers(0, 2, size=(stop - start, len(differences)), dtype=np.int8)
        signs = (signs.astype(float) * 2.0) - 1.0
        permuted = np.abs(np.mean(signs * differences, axis=1))
        at_least_as_extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return (at_least_as_extreme + 1) / (permutations + 1)


def mcnemar_exact(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
) -> dict[str, float | int]:
    """Exact two-sided McNemar test for paired binary outcomes."""
    values_a, values_b = paired_metric_values(records_a, records_b, metric_extractor)
    if not np.all(np.isin(values_a, (0.0, 1.0))) or not np.all(np.isin(values_b, (0.0, 1.0))):
        raise ValueError("McNemar exact test requires binary metric values encoded as 0 or 1.")
    a_correct_b_wrong = int(np.count_nonzero((values_a == 1.0) & (values_b == 0.0)))
    a_wrong_b_correct = int(np.count_nonzero((values_a == 0.0) & (values_b == 1.0)))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_correct_b_wrong, a_wrong_b_correct)
        lower_tail = Fraction(
            sum(math.comb(discordant, index) for index in range(lower + 1)),
            2**discordant,
        )
        p_value = min(1.0, float(2 * lower_tail))
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the caller's original order."""
    values = np.asarray(p_values, dtype=float)
    if not len(values):
        return []
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p-values must be finite and between 0 and 1.")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, float((len(values) - rank) * values[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return [float(value) for value in adjusted]


def relative_delta(reference_mean: float, comparison_mean: float) -> float | None:
    """Return comparison-minus-reference change relative to the reference mean."""
    if not math.isfinite(reference_mean) or not math.isfinite(comparison_mean):
        raise ValueError("Means must be finite.")
    if reference_mean == 0.0:
        return None
    return (comparison_mean - reference_mean) / abs(reference_mean)


def paired_standardized_effect_size(
    records_a: Sequence[PredictionRecordLike],
    records_b: Sequence[PredictionRecordLike],
    metric_extractor: MetricExtractor,
) -> float:
    values_a, values_b = paired_metric_values(records_a, records_b, metric_extractor)
    differences = values_a - values_b

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
    resamples: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    permutations: int = DEFAULT_PERMUTATION_REPLICATES,
    seed: int = DEFAULT_STATISTICS_SEED,
) -> dict[str, float | int | str]:
    values_a, values_b = paired_metric_values(records_a, records_b, metric_extractor)

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
        "n": len(values_a),
    }
