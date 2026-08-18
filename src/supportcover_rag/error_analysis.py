from __future__ import annotations

import random
import re
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from supportcover_rag.config import AppConfig
from supportcover_rag.io_utils import ensure_dir, read_jsonl, write_csv
from supportcover_rag.statistics import align_records_by_example_id
from supportcover_rag.text import informative_term_set, jaccard_similarity, normalize_answer

ABSTAIN_ANSWER = "insufficient evidence"
NUMERIC_TOKEN_RE = re.compile(r"\d+")
ERROR_CATEGORY_ORDER = [
    "support_missing",
    "support_present_answer_wrong",
    "formatting_mismatch",
    "hallucination",
    "multi_hop_reasoning_failure",
    "insufficient_evidence_forced_answer",
    "other",
]
ERROR_TAXONOMY = {
    "support_missing": "Gold-support evidence was not sufficiently preserved in the packed context.",
    "support_present_answer_wrong": "Evidence needed for the answer is present, but the model still produced the wrong answer.",
    "formatting_mismatch": "The prediction is semantically close but fails evaluation because of formatting, extra words, or output style.",
    "hallucination": "The model output is unsupported by the packed evidence.",
    "multi_hop_reasoning_failure": "Relevant evidence is partly present, but the model fails to connect the reasoning chain correctly.",
    "insufficient_evidence_forced_answer": "The packed evidence is insufficient, but the model gives a concrete answer instead of abstaining.",
    "other": "The failure does not fit the main taxonomy cleanly.",
}


@dataclass(slots=True)
class ErrorAnalysisSource:
    method: str
    run_dir: Path
    predictions_path: Path
    experiment_id: str
    model_alias: str


def _contains_normalized_answer(answer: str, evidence: str) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_evidence = normalize_answer(evidence)
    return bool(normalized_answer) and normalized_answer in normalized_evidence


def _same_numeric_sequence(left: str, right: str) -> bool:
    left_numbers = NUMERIC_TOKEN_RE.findall(left)
    right_numbers = NUMERIC_TOKEN_RE.findall(right)
    return bool(left_numbers) and left_numbers == right_numbers


def _is_formatting_mismatch(prediction: str, gold_answer: str, answer_f1: float) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    if not normalized_prediction or not normalized_gold:
        return False
    if answer_f1 >= 0.5:
        return True
    if normalized_prediction in normalized_gold or normalized_gold in normalized_prediction:
        return True
    if _same_numeric_sequence(prediction, gold_answer):
        return True
    return jaccard_similarity(
        informative_term_set(normalized_prediction),
        informative_term_set(normalized_gold),
    ) >= 0.8


def assign_error_category(record: dict[str, object]) -> str:
    predicted_answer = str(record.get("predicted_answer", ""))
    gold_answer = str(record.get("gold_answer", ""))
    answer_f1 = float(record.get("answer_f1", 0.0))
    coverage = float(record.get("coverage_at_budget", 0.0))
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    evidence_text = str(metadata.get("evidence_text", ""))
    question_type = str(metadata.get("question_type", ""))
    normalized_prediction = normalize_answer(predicted_answer)
    is_abstain = normalized_prediction == ABSTAIN_ANSWER
    gold_in_evidence = _contains_normalized_answer(gold_answer, evidence_text)
    pred_in_evidence = _contains_normalized_answer(predicted_answer, evidence_text)
    partial_support = 0.0 < coverage < 1.0
    full_support = coverage >= 0.75 or gold_in_evidence
    weak_support = coverage < 0.5 and not gold_in_evidence

    if not is_abstain and _is_formatting_mismatch(predicted_answer, gold_answer, answer_f1):
        return "formatting_mismatch"
    if is_abstain:
        if weak_support:
            return "support_missing"
        if full_support:
            return "support_present_answer_wrong"
        if partial_support:
            return "multi_hop_reasoning_failure"
        return "support_present_answer_wrong"
    if weak_support:
        return "insufficient_evidence_forced_answer"
    if question_type in {"bridge", "comparison"} and partial_support and not gold_in_evidence:
        return "multi_hop_reasoning_failure"
    if full_support:
        return "support_present_answer_wrong"
    if partial_support:
        return "multi_hop_reasoning_failure"
    if not pred_in_evidence and answer_f1 == 0.0:
        return "hallucination"
    return "other"


def _load_source(method: str, run_dir: str | Path) -> ErrorAnalysisSource:
    source_dir = Path(run_dir)
    predictions_path = source_dir / "predictions.jsonl"
    resolved_config_path = source_dir / "config.resolved.yaml"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions artifact: {predictions_path}")

    resolved_payload: dict[str, object] = {}
    if resolved_config_path.exists():
        resolved_payload = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    run_payload = resolved_payload.get("run", {}) if isinstance(resolved_payload, dict) else {}
    if not isinstance(run_payload, dict):
        run_payload = {}
    experiment_id = str(run_payload.get("experiment_id") or source_dir.name.split("_", 1)[0])
    model_alias = str(run_payload.get("model_alias") or source_dir.name.split("_")[2])
    return ErrorAnalysisSource(
        method=method,
        run_dir=source_dir,
        predictions_path=predictions_path,
        experiment_id=experiment_id,
        model_alias=model_alias,
    )


def _build_evidence_excerpt(evidence_text: str, max_chars: int = 420) -> str:
    compact = " ".join(line.strip() for line in evidence_text.splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _comparison_bucket(
    comparator_failed: bool,
    canonical_failed: bool,
    *,
    comparator_method: str,
    canonical_method: str,
) -> str:
    if comparator_failed and canonical_failed:
        return "both_fail"
    if comparator_failed:
        return f"{comparator_method}_fail_only"
    if canonical_failed:
        return f"{canonical_method}_fail_only"
    return "both_success"


def _build_note(
    *,
    category: str,
    record: dict[str, object],
    comparison_outcome: str,
    comparison_method: str,
) -> str:
    coverage = float(record.get("coverage_at_budget", 0.0))
    predicted_answer = str(record.get("predicted_answer", ""))
    category_notes = {
        "support_missing": f"Coverage is {coverage:.2f}; the packed context misses key support and the model abstains.",
        "support_present_answer_wrong": f"Answer evidence is present enough (coverage {coverage:.2f}), but the model still outputs '{predicted_answer}'.",
        "formatting_mismatch": f"The prediction overlaps with the gold answer but fails normalization-sensitive scoring.",
        "hallucination": "The prediction is not grounded in the packed evidence excerpt.",
        "multi_hop_reasoning_failure": f"Coverage is {coverage:.2f}; the chain is only partly preserved and the model fails to connect it.",
        "insufficient_evidence_forced_answer": f"Coverage is {coverage:.2f}; the model commits to a concrete answer instead of abstaining.",
        "other": "The failure does not fit the main taxonomy cleanly.",
    }
    if comparison_outcome == "both_fail":
        paired_note = f"The paired {comparison_method} run also fails on this example."
    else:
        paired_note = f"The paired {comparison_method} run succeeds on this example."
    return f"{category_notes[category]} {paired_note}"


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    category_rank = {name: index for index, name in enumerate(ERROR_CATEGORY_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            category_rank.get(str(row["assigned_error_category"]), len(ERROR_CATEGORY_ORDER)),
            str(row["example_id"]),
        ),
    )


def _round_robin_sample(rows: list[dict[str, object]], target: int) -> list[dict[str, object]]:
    grouped: dict[str, deque[dict[str, object]]] = {}
    for category in ERROR_CATEGORY_ORDER:
        grouped[category] = deque(_sort_rows([row for row in rows if row["assigned_error_category"] == category]))
    selected: list[dict[str, object]] = []
    while len(selected) < target:
        progressed = False
        for category in ERROR_CATEGORY_ORDER:
            if not grouped[category]:
                continue
            selected.append(grouped[category].popleft())
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def _build_annotation_rows(config: AppConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = config.error_analysis
    canonical_method = settings.canonical_method
    comparator_method = settings.comparator_method
    required_methods = [comparator_method, canonical_method]
    missing_methods = [method for method in required_methods if method not in settings.source_runs]
    if missing_methods:
        missing = ", ".join(missing_methods)
        raise ValueError(f"Missing source_runs entries for: {missing}")

    sources = {method: _load_source(method, settings.source_runs[method]) for method in required_methods}
    records_by_method = {
        method: {row["example_id"]: row for row in read_jsonl(source.predictions_path)}
        for method, source in sources.items()
    }
    if set(records_by_method[canonical_method]) != set(records_by_method[comparator_method]):
        raise ValueError("Phase 7 expects the same example ids in both frozen comparison runs.")

    failed_rows: list[dict[str, object]] = []
    for example_id in sorted(records_by_method[canonical_method]):
        canonical_record = records_by_method[canonical_method][example_id]
        comparator_record = records_by_method[comparator_method][example_id]
        comparison_outcome = _comparison_bucket(
            comparator_record["answer_em"] < 1.0,
            canonical_record["answer_em"] < 1.0,
            comparator_method=comparator_method,
            canonical_method=canonical_method,
        )
        for method in required_methods:
            record = records_by_method[method][example_id]
            if float(record["answer_em"]) >= 1.0:
                continue
            comparison_method = canonical_method if method == comparator_method else comparator_method
            comparison_record = records_by_method[comparison_method][example_id]
            source = sources[method]
            category = assign_error_category(record)
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            failed_rows.append(
                {
                    "example_id": example_id,
                    "method": method,
                    "model": source.model_alias,
                    "experiment_id": source.experiment_id,
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "predicted_answer": record["predicted_answer"],
                    "answer_correct": False,
                    "answer_f1": float(record["answer_f1"]),
                    "support_f1_example": float(record["support_f1"]),
                    "coverage_at_budget_example": float(record["coverage_at_budget"]),
                    "packed_evidence_excerpt": _build_evidence_excerpt(str(metadata.get("evidence_text", ""))),
                    "assigned_error_category": category,
                    "short_analyst_note": _build_note(
                        category=category,
                        record=record,
                        comparison_outcome=comparison_outcome,
                        comparison_method=comparison_method,
                    ),
                    "comparison_method": comparison_method,
                    "comparison_predicted_answer": comparison_record["predicted_answer"],
                    "comparison_answer_correct": bool(float(comparison_record["answer_em"]) >= 1.0),
                    "comparison_answer_f1": float(comparison_record["answer_f1"]),
                    "comparison_support_f1_example": float(comparison_record["support_f1"]),
                    "comparison_coverage_at_budget": float(comparison_record["coverage_at_budget"]),
                    "comparison_outcome": comparison_outcome,
                    "source_prediction_artifact": str(source.predictions_path),
                }
            )

    canonical_only = _sort_rows(
        [
            row
            for row in failed_rows
            if row["method"] == canonical_method and row["comparison_outcome"] == f"{canonical_method}_fail_only"
        ]
    )
    canonical_both_fail = _sort_rows(
        [row for row in failed_rows if row["method"] == canonical_method and row["comparison_outcome"] == "both_fail"]
    )
    comparator_only = _sort_rows(
        [
            row
            for row in failed_rows
            if row["method"] == comparator_method and row["comparison_outcome"] == f"{comparator_method}_fail_only"
        ]
    )
    comparator_both_fail = _sort_rows(
        [row for row in failed_rows if row["method"] == comparator_method and row["comparison_outcome"] == "both_fail"]
    )

    selected_annotations: list[dict[str, object]] = list(canonical_only)
    canonical_remaining = max(0, settings.canonical_sample_size - len(selected_annotations))
    selected_canonical_both_fail = _round_robin_sample(canonical_both_fail, canonical_remaining)
    selected_annotations.extend(selected_canonical_both_fail)

    selected_comparator: list[dict[str, object]] = list(comparator_only)
    comparator_remaining = max(0, settings.comparison_sample_size - len(selected_comparator))
    matched_ids = {str(row["example_id"]) for row in selected_canonical_both_fail}
    matched_comparator_rows = [row for row in comparator_both_fail if str(row["example_id"]) in matched_ids]
    selected_comparator.extend(matched_comparator_rows[:comparator_remaining])
    comparator_remaining = max(0, settings.comparison_sample_size - len(selected_comparator))
    if comparator_remaining:
        already_selected = {str(row["example_id"]) for row in selected_comparator}
        extra_rows = [row for row in comparator_both_fail if str(row["example_id"]) not in already_selected]
        selected_comparator.extend(_round_robin_sample(extra_rows, comparator_remaining))

    selected_annotations.extend(selected_comparator)
    return _sort_rows(selected_annotations), failed_rows


def _build_summary_rows(
    *,
    failed_rows: list[dict[str, object]],
    methods: list[str],
) -> list[dict[str, object]]:
    rows_by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in failed_rows:
        rows_by_method[str(row["method"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for method in methods:
        method_rows = rows_by_method[method]
        total_failures = len(method_rows)
        model = str(method_rows[0]["model"]) if method_rows else ""
        experiment_id = str(method_rows[0]["experiment_id"]) if method_rows else ""
        counts = Counter(str(row["assigned_error_category"]) for row in method_rows)
        for category in ERROR_CATEGORY_ORDER:
            count = counts.get(category, 0)
            summary_rows.append(
                {
                    "method": method,
                    "model": model,
                    "experiment_id": experiment_id,
                    "error_category": category,
                    "count": count,
                    "failure_share": 0.0 if total_failures == 0 else count / total_failures,
                    "total_failures": total_failures,
                }
            )
    return summary_rows


def _select_representative_examples(
    annotations: list[dict[str, object]],
    *,
    canonical_method: str,
    comparator_method: str,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    def maybe_add(predicate: object) -> None:
        nonlocal selected
        if len(selected) >= limit:
            return
        for row in annotations:
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                continue
            if predicate(row):
                selected.append(row)
                seen_ids.add(example_id)
                return

    maybe_add(lambda row: row["method"] == comparator_method and row["comparison_outcome"] == f"{comparator_method}_fail_only")
    maybe_add(lambda row: row["method"] == canonical_method and row["comparison_outcome"] == f"{canonical_method}_fail_only")
    maybe_add(lambda row: row["method"] == canonical_method and row["assigned_error_category"] == "support_missing")
    maybe_add(lambda row: row["method"] == canonical_method and row["assigned_error_category"] == "multi_hop_reasoning_failure")
    maybe_add(lambda row: row["method"] == canonical_method and row["assigned_error_category"] == "support_present_answer_wrong")

    if len(selected) < limit:
        for row in annotations:
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(example_id)
            if len(selected) >= limit:
                break
    return selected


def _blinded_evidence(record: Mapping[str, object], example_id: str) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Prediction {example_id} is missing metadata required for blinded evidence review.")
    evidence = metadata.get("evidence_text")
    if not isinstance(evidence, str):
        raise ValueError(f"Prediction {example_id} is missing string metadata.evidence_text.")
    return evidence


def _required_blinded_field(record: Mapping[str, object], field: str, example_id: str) -> object:
    if field not in record:
        raise ValueError(f"Prediction {example_id} is missing required field '{field}'.")
    return record[field]


def build_blinded_annotation_records(
    comparator_predictions: Sequence[Mapping[str, object]],
    supportcover_predictions: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Build reproducibly randomized A/B rows without exposing system identity."""
    aligned = align_records_by_example_id(comparator_predictions, supportcover_predictions)
    rng = random.Random(seed)
    annotations: list[dict[str, object]] = []
    for comparator_record, supportcover_record in aligned:
        example_id = str(comparator_record["example_id"])
        for field in ("question", "gold_answer", "gold_supporting_facts"):
            comparator_value = _required_blinded_field(comparator_record, field, example_id)
            supportcover_value = _required_blinded_field(supportcover_record, field, example_id)
            if comparator_value != supportcover_value:
                raise ValueError(f"Prediction metadata mismatch for example {example_id}: {field}")

        _required_blinded_field(comparator_record, "predicted_answer", example_id)
        _required_blinded_field(supportcover_record, "predicted_answer", example_id)

        if rng.getrandbits(1):
            record_a, record_b = supportcover_record, comparator_record
        else:
            record_a, record_b = comparator_record, supportcover_record
        annotations.append(
            {
                "example_id": example_id,
                "question": _required_blinded_field(comparator_record, "question", example_id),
                "gold_answer": _required_blinded_field(comparator_record, "gold_answer", example_id),
                "gold_support": _required_blinded_field(comparator_record, "gold_supporting_facts", example_id),
                "evidence_a": _blinded_evidence(record_a, example_id),
                "evidence_b": _blinded_evidence(record_b, example_id),
                "prediction_a": _required_blinded_field(record_a, "predicted_answer", example_id),
                "prediction_b": _required_blinded_field(record_b, "predicted_answer", example_id),
                "preferred_evidence": "",
                "preferred_prediction": "",
                "error_category_a": "",
                "error_category_b": "",
                "annotator_notes": "",
            }
        )
    return annotations


def _render_analysis_markdown(
    *,
    config: AppConfig,
    summary_rows: list[dict[str, object]],
    annotations: list[dict[str, object]],
) -> str:
    settings = config.error_analysis
    canonical_method = settings.canonical_method
    comparator_method = settings.comparator_method
    summary_lookup = {
        (str(row["method"]), str(row["error_category"])): row
        for row in summary_rows
    }
    representative_rows = _select_representative_examples(
        annotations,
        canonical_method=canonical_method,
        comparator_method=comparator_method,
        limit=settings.representative_examples,
    )

    lines: list[str] = [
        "# Phase 7 Error Analysis",
        "",
        "Phase 7 reuses the frozen Qwen comparison from Phase 6 and analyzes only `relevance_only` versus `supportcover_final`, where `supportcover_final = no_redundancy`.",
        "",
        f"Frozen setup config: `{settings.frozen_setup_config}`",
        f"- comparator source: `{settings.source_runs[comparator_method]}`",
        f"- canonical source: `{settings.source_runs[canonical_method]}`",
        "",
        "Taxonomy:",
    ]
    for category in settings.taxonomy:
        lines.append(f"- `{category}`: {ERROR_TAXONOMY[category]}")

    lines.extend(
        [
            "",
            "Sampling protocol:",
            f"- annotated failed rows: `{len(annotations)}`",
            f"- supportcover_final focus rows: `{settings.canonical_sample_size}`",
            f"- relevance_only comparison rows: `{settings.comparison_sample_size}`",
            "",
            "| category | relevance_only | supportcover_final |",
            "| --- | --- | --- |",
        ]
    )
    for category in ERROR_CATEGORY_ORDER:
        comparator_row = summary_lookup[(comparator_method, category)]
        canonical_row = summary_lookup[(canonical_method, category)]
        comparator_text = f"{comparator_row['count']} ({float(comparator_row['failure_share']):.1%})"
        canonical_text = f"{canonical_row['count']} ({float(canonical_row['failure_share']):.1%})"
        lines.append(f"| {category} | {comparator_text} | {canonical_text} |")

    lines.extend(
        [
            "",
            "Representative examples:",
            "| example_id | method | category | explanation |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in representative_rows:
        explanation = str(row["short_analyst_note"]).replace("|", "\\|")
        lines.append(
            f"| {row['example_id']} | {row['method']} | {row['assigned_error_category']} | {explanation} |"
        )

    comparator_missing = int(summary_lookup[(comparator_method, "support_missing")]["count"])
    canonical_missing = int(summary_lookup[(canonical_method, "support_missing")]["count"])
    comparator_present_wrong = int(summary_lookup[(comparator_method, "support_present_answer_wrong")]["count"])
    canonical_present_wrong = int(summary_lookup[(canonical_method, "support_present_answer_wrong")]["count"])
    comparator_multi_hop = int(summary_lookup[(comparator_method, "multi_hop_reasoning_failure")]["count"])
    canonical_multi_hop = int(summary_lookup[(canonical_method, "multi_hop_reasoning_failure")]["count"])
    comparator_formatting = int(summary_lookup[(comparator_method, "formatting_mismatch")]["count"])
    canonical_formatting = int(summary_lookup[(canonical_method, "formatting_mismatch")]["count"])
    comparator_hallucination = int(summary_lookup[(comparator_method, "hallucination")]["count"])
    canonical_hallucination = int(summary_lookup[(canonical_method, "hallucination")]["count"])

    lines.extend(
        [
            "",
            "Main read:",
            f"- `supportcover_final` reduces `support_missing` from `{comparator_missing}` to `{canonical_missing}`, which is consistent with the earlier coverage gains.",
            f"- The largest remaining bucket for both methods is `support_present_answer_wrong` (`{comparator_present_wrong}` for `relevance_only`, `{canonical_present_wrong}` for `supportcover_final`), which suggests the bottleneck shifts toward generation once evidence is preserved.",
            f"- `multi_hop_reasoning_failure` remains common (`{comparator_multi_hop}` vs `{canonical_multi_hop}`), so better support coverage does not fully solve reasoning-chain failures.",
            f"- `formatting_mismatch` is visible but secondary (`{comparator_formatting}` vs `{canonical_formatting}`); it affects a handful of answer misses but does not dominate the error profile.",
            f"- `hallucination` is not a major issue under this conservative labeling pass (`{comparator_hallucination}` for `relevance_only`, `{canonical_hallucination}` for `supportcover_final`).",
        ]
    )
    if int(summary_lookup[(comparator_method, "insufficient_evidence_forced_answer")]["count"]) or int(
        summary_lookup[(canonical_method, "insufficient_evidence_forced_answer")]["count"]
    ):
        lines.append(
            "- `insufficient_evidence_forced_answer` is rare, which suggests the abstention instruction is usually respected when support is clearly missing."
        )

    return "\n".join(lines) + "\n"


def run_error_analysis(config: AppConfig) -> dict[str, Path]:
    settings = config.error_analysis
    unknown_categories = [category for category in settings.taxonomy if category not in ERROR_TAXONOMY]
    if unknown_categories:
        unknown = ", ".join(unknown_categories)
        raise ValueError(f"Unknown error taxonomy categories: {unknown}")

    annotations, failed_rows = _build_annotation_rows(config)
    output_dir = ensure_dir(settings.output_dir)
    annotation_path = output_dir / "phase7_error_annotations.csv"
    summary_path = output_dir / "phase7_error_summary.csv"
    analysis_path = output_dir / "phase7_error_analysis.md"

    write_csv(annotation_path, annotations)
    summary_rows = _build_summary_rows(
        failed_rows=failed_rows,
        methods=[settings.comparator_method, settings.canonical_method],
    )
    write_csv(summary_path, summary_rows)
    analysis_path.write_text(
        _render_analysis_markdown(config=config, summary_rows=summary_rows, annotations=annotations),
        encoding="utf-8",
    )
    return {
        "annotation_path": annotation_path,
        "summary_path": summary_path,
        "analysis_path": analysis_path,
    }
