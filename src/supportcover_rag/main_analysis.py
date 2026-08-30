from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from supportcover_rag.evaluation import aggregate_records
from supportcover_rag.final_validation import FairComparisonValidation, validate_fair_comparison
from supportcover_rag.statistics import align_records_by_example_id, build_comparison_row
from supportcover_rag.types import PredictionRecord


SUPPORTCOVER_METHOD = "supportcover_final"
PAIRED_COMPARATORS = (
    "paragraph_topk",
    "relevance_only",
    "mmr_sentence",
    "greedy_query_cover",
    "external_compressor",
)
DEFAULT_PAIRED_METRICS = (
    "answer_em",
    "answer_f1",
    "support_f1",
    "support_recall",
    "coverage_at_budget",
)


@dataclass(frozen=True, slots=True)
class MainAnalysisResult:
    aggregates: dict[str, dict[str, float | int | str | None]]
    comparisons: list[dict[str, float | int | str]]
    fair_comparison: FairComparisonValidation | None = None


def _prediction_mapping(record: Mapping[str, Any] | PredictionRecord) -> dict[str, Any]:
    if isinstance(record, PredictionRecord):
        return record.to_dict()
    return dict(record)


def _prediction_record(record: Mapping[str, Any]) -> PredictionRecord:
    try:
        return PredictionRecord(**dict(record))
    except TypeError as exc:
        example_id = record.get("example_id", "unknown")
        raise ValueError(f"Malformed prediction record for example {example_id}: {exc}") from exc


def _metric_extractor(metric: str):
    def extract(record: Mapping[str, Any]) -> float:
        if metric not in record:
            raise ValueError(f"Prediction record {record.get('example_id', 'unknown')} is missing metric '{metric}'.")
        return float(record[metric])

    return extract


def _metric_available(records: Sequence[Mapping[str, Any]], metric: str) -> bool:
    availability = [record.get(metric) is not None for record in records]
    if any(availability) and not all(availability):
        raise ValueError(f"Metric '{metric}' is inconsistently available within one method population.")
    return all(availability)


def analyze_main_predictions(
    predictions_by_method: Mapping[str, Sequence[Mapping[str, Any] | PredictionRecord]],
    *,
    metrics: Sequence[str] = DEFAULT_PAIRED_METRICS,
    resolved_configs: Mapping[str, Mapping[str, Any]] | None = None,
    resamples: int = 10_000,
    confidence: float = 0.95,
    permutations: int = 10_000,
    seed: int = 42,
) -> MainAnalysisResult:
    """Historical in-memory aggregation; publication artifacts use run-statistics."""
    missing_methods = [
        method
        for method in (SUPPORTCOVER_METHOD, *PAIRED_COMPARATORS)
        if method not in predictions_by_method
    ]
    if missing_methods:
        raise ValueError(f"Missing main-study prediction methods: {', '.join(missing_methods)}")
    if not predictions_by_method:
        raise ValueError("Main analysis requires prediction records.")

    normalized: dict[str, list[dict[str, Any]]] = {
        method: [_prediction_mapping(record) for record in records]
        for method, records in predictions_by_method.items()
    }
    empty_methods = [method for method, records in normalized.items() if not records]
    if empty_methods:
        raise ValueError(f"Main-study methods have no prediction records: {', '.join(empty_methods)}")

    reference_method = next(iter(normalized))
    reference_records = normalized[reference_method]
    for method, records in normalized.items():
        if method != reference_method:
            align_records_by_example_id(reference_records, records)

    fair_comparison = None
    if resolved_configs is not None:
        missing_configs = [method for method in normalized if method not in resolved_configs]
        if missing_configs:
            raise ValueError(f"Missing resolved configs for methods: {', '.join(missing_configs)}")
        fair_comparison = validate_fair_comparison(
            {method: resolved_configs[method] for method in normalized}
        )
    aggregates = {
        method: aggregate_records([_prediction_record(record) for record in records])
        for method, records in normalized.items()
    }

    supportcover_records = normalized[SUPPORTCOVER_METHOD]
    comparisons: list[dict[str, float | int | str]] = []
    for comparator in PAIRED_COMPARATORS:
        comparator_records = normalized[comparator]
        for metric in metrics:
            if not _metric_available(supportcover_records, metric) or not _metric_available(
                comparator_records,
                metric,
            ):
                continue
            comparisons.append(
                build_comparison_row(
                    metric=metric,
                    method_a=SUPPORTCOVER_METHOD,
                    method_b=comparator,
                    records_a=supportcover_records,
                    records_b=comparator_records,
                    metric_extractor=_metric_extractor(metric),
                    resamples=resamples,
                    confidence=confidence,
                    permutations=permutations,
                    seed=seed,
                )
            )

    return MainAnalysisResult(
        aggregates=aggregates,
        comparisons=comparisons,
        fair_comparison=fair_comparison,
    )
