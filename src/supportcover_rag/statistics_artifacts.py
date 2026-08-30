from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from supportcover_rag.io_utils import ensure_dir, read_jsonl, write_csv, write_json
from supportcover_rag.statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_CONFIDENCE,
    DEFAULT_PERMUTATION_REPLICATES,
    DEFAULT_STATISTICS_SEED,
    align_records_by_example_id,
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap,
    paired_metric_values,
    paired_random_sign_test,
    relative_delta,
)


STATISTICS_SCHEMA_VERSION = 1
DEFAULT_METRICS = (
    "answer_em",
    "answer_f1",
    "support_f1",
    "support_recall",
    "coverage_at_budget",
)
OPTIONAL_METRICS = (
    "support_precision",
    "evidence_tokens",
    "retrieval_latency_ms",
    "packing_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
)
SUPPORTED_METRICS = DEFAULT_METRICS + OPTIONAL_METRICS
_PROTOCOL_FIELDS = (
    "dataset",
    "model",
    "prompt_settings",
    "decoding_settings",
    "token_budget",
    "retrieval_depth",
    "retrieval_settings",
)


@dataclass(slots=True)
class LoadedStatisticsRun:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path
    config_path: Path
    method: str
    config_id: str
    run_id: str
    records: list[dict[str, Any]]
    run_metrics: dict[str, Any]
    protocol: dict[str, Any]
    source_hashes: dict[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _split_alias(value: object) -> str:
    split = _require_nonempty_string(value, name="split").lower()
    aliases = {"validation": "val", "valid": "val", "val": "val", "train": "train", "test": "test"}
    return aliases.get(split, split)


def _resolve_path(value: object, *, project_root: Path, name: str) -> Path:
    raw = _require_nonempty_string(value, name=name)
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(_require_mapping(payload, name=name))


def _load_yaml_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(_require_mapping(payload, name=name))


def _protocol_from_config(config: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    raw_data = _require_mapping(config.get("raw_data"), name="config.raw_data")
    generation = _require_mapping(config.get("generation"), name="config.generation")
    prompting = _require_mapping(config.get("prompting"), name="config.prompting")
    retrieval = _require_mapping(config.get("retrieval"), name="config.retrieval")
    _require_mapping(config.get("supportcover"), name="config.supportcover")
    return {
        "dataset": {
            "path": raw_data.get("dataset_path"),
            "config": raw_data.get("dataset_config"),
        },
        "model": generation.get("model_name_or_path"),
        "prompt_settings": dict(prompting),
        "decoding_settings": {
            key: generation.get(key)
            for key in ("backend", "temperature", "max_new_tokens", "do_sample")
        },
        "token_budget": run.get("token_budget"),
        "retrieval_depth": run.get("retrieval_depth"),
        "retrieval_settings": {
            key: retrieval.get(key)
            for key in ("method", "bm25_k1", "bm25_b")
        },
    }


def _validate_metric_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    source_name: str,
) -> None:
    for row_index, record in enumerate(records, start=1):
        for metric in metrics:
            if metric not in record:
                raise ValueError(f"{source_name} row {row_index} is missing metric '{metric}'.")
            try:
                value = float(record[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source_name} row {row_index} metric '{metric}' is not numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{source_name} row {row_index} metric '{metric}' must be finite.")
            if metric == "answer_em" and value not in (0.0, 1.0):
                raise ValueError(f"{source_name} row {row_index} answer_em must be binary (0 or 1).")


def _load_run(
    descriptor: Mapping[str, Any],
    *,
    project_root: Path,
    expected_dataset: str,
    expected_split: str,
    expected_role: str,
    expected_split_sha256: str,
    expected_num_examples: int,
    metrics: Sequence[str],
) -> LoadedStatisticsRun:
    run_dir = _resolve_path(descriptor.get("run_dir"), project_root=project_root, name="run_dir")
    method = _require_nonempty_string(descriptor.get("method"), name="method")
    config_id = _require_nonempty_string(descriptor.get("config_id"), name="config_id")
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.resolved.yaml"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions artifact: {predictions_path}")
    records = read_jsonl(predictions_path)
    run_metrics = _load_json_object(metrics_path, name="run metrics")
    config = _load_yaml_object(config_path, name="resolved run config")
    config_run = _require_mapping(config.get("run"), name="config.run")
    split_config = _require_mapping(config.get("split"), name="config.split")
    experiments = _require_mapping(config.get("experiments"), name="config.experiments")

    checks = {
        "status": (run_metrics.get("status"), "completed"),
        "dataset": (run_metrics.get("dataset"), expected_dataset),
        "split": (_split_alias(run_metrics.get("split")), _split_alias(expected_split)),
        "split SHA256": (run_metrics.get("split_sha256"), expected_split_sha256),
        "method": (run_metrics.get("method"), method),
        "number of examples": (run_metrics.get("num_examples"), expected_num_examples),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{run_dir} {field} mismatch: expected {expected!r}, found {actual!r}.")
    if split_config.get("role") != expected_role:
        raise ValueError(
            f"{run_dir} scientific-role mismatch: expected {expected_role!r}, found {split_config.get('role')!r}."
        )
    if _split_alias(experiments.get("split")) != _split_alias(expected_split):
        raise ValueError(f"{run_dir} resolved config uses a different experiment split.")
    for field in ("experiment_id", "method", "dataset", "split", "token_budget", "retrieval_depth", "split_sha256"):
        if config_run.get(field) != run_metrics.get(field):
            raise ValueError(f"{run_dir} metrics/config provenance mismatch for '{field}'.")
    if len(records) != expected_num_examples:
        raise ValueError(
            f"{run_dir} prediction count mismatch: expected {expected_num_examples}, found {len(records)}."
        )
    for row_index, record in enumerate(records, start=1):
        if record.get("method") != method:
            raise ValueError(f"{run_dir} prediction row {row_index} has an unexpected method.")
        if record.get("token_budget") != run_metrics.get("token_budget"):
            raise ValueError(f"{run_dir} prediction row {row_index} has an unexpected token budget.")
    _validate_metric_records(records, metrics=metrics, source_name=str(predictions_path))
    # Indexing a source against itself validates IDs even before pairwise comparison.
    align_records_by_example_id(records, records)

    return LoadedStatisticsRun(
        run_dir=run_dir,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        config_path=config_path,
        method=method,
        config_id=config_id,
        run_id=_require_nonempty_string(run_metrics.get("experiment_id"), name="experiment_id"),
        records=records,
        run_metrics=run_metrics,
        protocol=_protocol_from_config(config, config_run),
        source_hashes={
            "predictions_sha256": _sha256_file(predictions_path),
            "metrics_sha256": _sha256_file(metrics_path),
            "resolved_config_sha256": _sha256_file(config_path),
        },
    )


def _validate_protocol(reference: LoadedStatisticsRun, comparison: LoadedStatisticsRun) -> None:
    mismatches = [
        field
        for field in _PROTOCOL_FIELDS
        if reference.protocol.get(field) != comparison.protocol.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Runs do not share the required evaluation protocol: " + ", ".join(mismatches)
        )


def _extract_metric(metric: str):
    def extractor(record: Mapping[str, Any]) -> float:
        return float(record[metric])

    return extractor


def _method_summary_rows(
    runs: Sequence[LoadedStatisticsRun],
    *,
    dataset: str,
    split: str,
    role: str,
    split_sha256: str,
    metrics: Sequence[str],
    bootstrap_replicates: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for metric in metrics:
            extractor = _extract_metric(metric)
            values = np.asarray([extractor(record) for record in run.records], dtype=float)
            ci_low, ci_high = bootstrap_ci(
                run.records,
                extractor,
                resamples=bootstrap_replicates,
                confidence=confidence,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "role": role,
                    "split_sha256": split_sha256,
                    "method": run.method,
                    "config_id": run.config_id,
                    "num_examples": len(run.records),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence_level": confidence,
                    "bootstrap_replicates": bootstrap_replicates,
                    "seed": seed,
                    "source_run_id": run.run_id,
                    "source_predictions": str(run.predictions_path),
                }
            )
    return rows


def _comparison_rows(
    reference: LoadedStatisticsRun,
    comparisons: Sequence[LoadedStatisticsRun],
    *,
    dataset: str,
    split: str,
    role: str,
    split_sha256: str,
    comparison_family: str,
    metrics: Sequence[str],
    bootstrap_replicates: int,
    permutation_replicates: int,
    confidence: float,
    seed: int,
    apply_holm: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        _validate_protocol(reference, comparison)
        align_records_by_example_id(reference.records, comparison.records)
        for metric in metrics:
            extractor = _extract_metric(metric)
            reference_values, comparison_values = paired_metric_values(
                reference.records,
                comparison.records,
                extractor,
            )
            reference_mean = float(np.mean(reference_values))
            comparison_mean = float(np.mean(comparison_values))
            absolute_delta = comparison_mean - reference_mean
            # paired_bootstrap reports A-B, so comparison is intentionally method A.
            delta_ci_low, delta_ci_high = paired_bootstrap(
                comparison.records,
                reference.records,
                extractor,
                resamples=bootstrap_replicates,
                confidence=confidence,
                seed=seed,
            )
            if metric == "answer_em":
                result = mcnemar_exact(comparison.records, reference.records, extractor)
                test = "mcnemar_exact_two_sided"
                statistic: float | str = ""
                p_value = float(result["p_value"])
                comparison_correct_reference_wrong = int(result["a_correct_b_wrong"])
                comparison_wrong_reference_correct = int(result["a_wrong_b_correct"])
            else:
                test = "paired_random_sign_two_sided"
                statistic = abs(absolute_delta)
                p_value = paired_random_sign_test(
                    comparison.records,
                    reference.records,
                    extractor,
                    permutations=permutation_replicates,
                    seed=seed,
                )
                comparison_correct_reference_wrong = ""
                comparison_wrong_reference_correct = ""
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "role": role,
                    "split_sha256": split_sha256,
                    "comparison_family": comparison_family,
                    "reference_method": reference.method,
                    "reference_config_id": reference.config_id,
                    "comparison_method": comparison.method,
                    "comparison_config_id": comparison.config_id,
                    "num_examples": len(reference_values),
                    "metric": metric,
                    "reference_mean": reference_mean,
                    "comparison_mean": comparison_mean,
                    "absolute_delta": absolute_delta,
                    "relative_delta": relative_delta(reference_mean, comparison_mean),
                    "delta_ci_low": delta_ci_low,
                    "delta_ci_high": delta_ci_high,
                    "test": test,
                    "statistic": statistic,
                    "p_value": p_value,
                    "p_value_adjusted": "",
                    "significant_at_0_05": "",
                    "comparison_correct_reference_wrong": comparison_correct_reference_wrong,
                    "comparison_wrong_reference_correct": comparison_wrong_reference_correct,
                    "absolute_percentage_point_change": 100.0 * absolute_delta if metric == "answer_em" else "",
                    "bootstrap_replicates": bootstrap_replicates,
                    "permutation_replicates": permutation_replicates if metric != "answer_em" else "",
                    "seed": seed,
                    "reference_run_id": reference.run_id,
                    "comparison_run_id": comparison.run_id,
                    "reference_predictions": str(reference.predictions_path),
                    "comparison_predictions": str(comparison.predictions_path),
                }
            )
    adjusted = holm_bonferroni([float(row["p_value"]) for row in rows]) if apply_holm else None
    for index, row in enumerate(rows):
        decision_p = adjusted[index] if adjusted is not None else float(row["p_value"])
        row["p_value_adjusted"] = adjusted[index] if adjusted is not None else ""
        row["significant_at_0_05"] = decision_p < 0.05
    return rows


def run_statistics_plan(
    plan_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    allow_nonpublication_replicates: bool = False,
    allow_unfrozen_for_tests: bool = False,
) -> dict[str, Any]:
    """Validate paired runs, compute statistics, and write raw reproducibility artifacts."""
    plan_source = Path(plan_path)
    plan = _load_json_object(plan_source, name="statistics plan")
    if plan.get("schema_version") != STATISTICS_SCHEMA_VERSION:
        raise ValueError(f"statistics plan schema_version must be {STATISTICS_SCHEMA_VERSION}.")
    project_root_value = plan.get("project_root", ".")
    project_root = _resolve_path(project_root_value, project_root=Path.cwd(), name="project_root")
    freeze_manifest_path: Path | None = None
    freeze_manifest_sha256: str | None = None
    freeze_manifest: dict[str, Any] | None = None
    if not allow_unfrozen_for_tests:
        freeze_manifest_path = _resolve_path(
            plan.get("phase3_freeze_manifest"),
            project_root=project_root,
            name="phase3_freeze_manifest",
        )
        freeze_manifest = _load_json_object(freeze_manifest_path, name="Phase-3 freeze manifest")
        if freeze_manifest.get("selection_role") != "development_only":
            raise ValueError("Phase-3 freeze manifest must record selection_role='development_only'.")
        if freeze_manifest.get("final_predictions_inspected") is not False:
            raise ValueError("Phase-3 freeze manifest must record final_predictions_inspected=false.")
        freeze_manifest_sha256 = _sha256_file(freeze_manifest_path)
    dataset = _require_nonempty_string(plan.get("dataset"), name="dataset")
    split = _require_nonempty_string(plan.get("split"), name="split")
    role = _require_nonempty_string(plan.get("role"), name="role").lower()
    if role not in {"development", "final"}:
        raise ValueError("role must be development or final.")
    split_sha256 = _require_nonempty_string(plan.get("split_sha256"), name="split_sha256")
    if len(split_sha256) != 64 or any(char not in "0123456789abcdef" for char in split_sha256.lower()):
        raise ValueError("split_sha256 must be a 64-character hexadecimal SHA256 digest.")
    if freeze_manifest is not None:
        freeze_split_field = "development_split_sha256" if role == "development" else "final_split_sha256"
        if freeze_manifest.get(freeze_split_field) != split_sha256:
            raise ValueError(f"Statistics split SHA256 does not match freeze manifest field '{freeze_split_field}'.")
    num_examples = plan.get("num_examples")
    if not isinstance(num_examples, int) or isinstance(num_examples, bool) or num_examples <= 0:
        raise ValueError("num_examples must be a positive integer.")
    comparison_family = _require_nonempty_string(plan.get("comparison_family"), name="comparison_family")
    metrics_value = plan.get("metrics", list(DEFAULT_METRICS))
    if not isinstance(metrics_value, list) or not metrics_value:
        raise ValueError("metrics must be a non-empty list.")
    metrics = [_require_nonempty_string(value, name="metric") for value in metrics_value]
    if len(set(metrics)) != len(metrics):
        raise ValueError("metrics must not contain duplicates.")
    unknown_metrics = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if unknown_metrics:
        raise ValueError("Unsupported statistics metrics: " + ", ".join(unknown_metrics))

    bootstrap_replicates = int(plan.get("bootstrap_replicates", DEFAULT_BOOTSTRAP_REPLICATES))
    permutation_replicates = int(plan.get("permutation_replicates", DEFAULT_PERMUTATION_REPLICATES))
    if bootstrap_replicates <= 0 or permutation_replicates <= 0:
        raise ValueError("Bootstrap and permutation replicate counts must be positive.")
    if not allow_nonpublication_replicates and (
        bootstrap_replicates < DEFAULT_BOOTSTRAP_REPLICATES
        or permutation_replicates < DEFAULT_PERMUTATION_REPLICATES
    ):
        raise ValueError("Publication statistics require at least 10,000 bootstrap and permutation replicates.")
    confidence = float(plan.get("confidence_level", DEFAULT_CONFIDENCE))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    seed = int(plan.get("seed", DEFAULT_STATISTICS_SEED))
    apply_holm = bool(plan.get("holm_bonferroni", False))

    reference_descriptor = _require_mapping(plan.get("reference"), name="reference")
    comparisons_value = plan.get("comparisons")
    if not isinstance(comparisons_value, list) or not comparisons_value:
        raise ValueError("comparisons must be a non-empty list.")
    comparison_descriptors = [
        _require_mapping(value, name=f"comparisons[{index}]")
        for index, value in enumerate(comparisons_value)
    ]
    reference = _load_run(
        reference_descriptor,
        project_root=project_root,
        expected_dataset=dataset,
        expected_split=split,
        expected_role=role,
        expected_split_sha256=split_sha256,
        expected_num_examples=num_examples,
        metrics=metrics,
    )
    comparisons = [
        _load_run(
            descriptor,
            project_root=project_root,
            expected_dataset=dataset,
            expected_split=split,
            expected_role=role,
            expected_split_sha256=split_sha256,
            expected_num_examples=num_examples,
            metrics=metrics,
        )
        for descriptor in comparison_descriptors
    ]
    config_ids = [reference.config_id] + [comparison.config_id for comparison in comparisons]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("reference/comparison config_id values must be unique.")

    method_rows = _method_summary_rows(
        [reference, *comparisons],
        dataset=dataset,
        split=split,
        role=role,
        split_sha256=split_sha256,
        metrics=metrics,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
        seed=seed,
    )
    comparison_rows = _comparison_rows(
        reference,
        comparisons,
        dataset=dataset,
        split=split,
        role=role,
        split_sha256=split_sha256,
        comparison_family=comparison_family,
        metrics=metrics,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        confidence=confidence,
        seed=seed,
        apply_holm=apply_holm,
    )

    target_dir = Path(output_dir) if output_dir is not None else project_root / "outputs" / "statistics"
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Statistics output directory is not empty: {target_dir}")
    ensure_dir(target_dir)
    method_summary_path = target_dir / "method_summary.csv"
    paired_comparisons_path = target_dir / "paired_comparisons.csv"
    manifest_path = target_dir / "statistics_manifest.json"
    write_csv(method_summary_path, method_rows)
    write_csv(paired_comparisons_path, comparison_rows)

    sources = []
    for run in (reference, *comparisons):
        sources.append(
            {
                "run_id": run.run_id,
                "method": run.method,
                "config_id": run.config_id,
                "run_dir": str(run.run_dir),
                "predictions": str(run.predictions_path),
                "metrics": str(run.metrics_path),
                "resolved_config": str(run.config_path),
                **run.source_hashes,
            }
        )
    manifest = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": split,
        "role": role,
        "split_sha256": split_sha256,
        "num_examples": num_examples,
        "metrics": metrics,
        "comparison_family": comparison_family,
        "holm_bonferroni": apply_holm,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "confidence_level": confidence,
            "method": "paired percentile bootstrap for deltas; percentile bootstrap for method means",
        },
        "permutation": {
            "replicates": permutation_replicates,
            "alternative": "two-sided",
            "method": "paired random-sign test",
        },
        "answer_em_test": "McNemar exact two-sided",
        "seed": seed,
        "code_revision": plan.get("code_revision"),
        "phase3_freeze_manifest": (
            {"path": str(freeze_manifest_path), "sha256": freeze_manifest_sha256}
            if freeze_manifest_path is not None
            else None
        ),
        "plan": {"path": str(plan_source), "sha256": _sha256_file(plan_source)},
        "source_artifacts": sources,
        "artifacts": {
            "method_summary": {
                "path": str(method_summary_path),
                "sha256": _sha256_file(method_summary_path),
            },
            "paired_comparisons": {
                "path": str(paired_comparisons_path),
                "sha256": _sha256_file(paired_comparisons_path),
            },
        },
    }
    write_json(manifest_path, manifest)
    return {
        "method_summary": str(method_summary_path),
        "paired_comparisons": str(paired_comparisons_path),
        "statistics_manifest": str(manifest_path),
        "statistics_manifest_sha256": _sha256_file(manifest_path),
    }
