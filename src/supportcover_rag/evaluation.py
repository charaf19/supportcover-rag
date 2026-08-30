from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from supportcover_rag.text import normalize_answer
from supportcover_rag.types import PredictionRecord, SupportKey


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def support_metrics(predicted: Iterable[SupportKey], gold: Iterable[SupportKey]) -> dict[str, float]:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not pred_set and not gold_set:
        return {"support_em": 1.0, "support_precision": 1.0, "support_recall": 1.0, "support_f1": 1.0}
    if not pred_set:
        return {"support_em": 0.0, "support_precision": 0.0, "support_recall": 0.0, "support_f1": 0.0}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    em = float(pred_set == gold_set)
    return {
        "support_em": em,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def coverage_at_budget(predicted: Iterable[SupportKey], gold: Iterable[SupportKey]) -> float:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    return len(pred_set & gold_set) / len(gold_set)


def aggregate_records(records: list[PredictionRecord]) -> dict[str, float | int | str | None]:
    if not records:
        return {"num_examples": 0}
    return {
        "method": records[0].method,
        "token_budget": records[0].token_budget,
        "num_examples": len(records),
        "answer_em": mean(r.answer_em for r in records),
        "answer_f1": mean(r.answer_f1 for r in records),
        "support_em": _optional_mean([r.support_em for r in records]),
        "support_precision": _optional_mean([r.support_precision for r in records]),
        "support_recall": _optional_mean([r.support_recall for r in records]),
        "support_f1": _optional_mean([r.support_f1 for r in records]),
        "coverage_at_budget": _optional_mean([r.coverage_at_budget for r in records]),
        "evidence_tokens": mean(r.evidence_tokens for r in records),
        "retrieval_latency_ms": mean(r.retrieval_latency_ms for r in records),
        "packing_latency_ms": mean(r.packing_latency_ms for r in records),
        "generation_latency_ms": mean(r.generation_latency_ms for r in records),
        "total_latency_ms": mean(r.total_latency_ms for r in records),
        "peak_rss_mb": mean(r.peak_rss_mb for r in records),
    }


def _optional_mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    if len(available) != len(values):
        raise ValueError("Metric availability must be consistent across a prediction population.")
    return mean(available)
