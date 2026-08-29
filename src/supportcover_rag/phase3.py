from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from supportcover_rag.config import AppConfig, SupportCoverConfig
from supportcover_rag.data import load_examples_by_ids
from supportcover_rag.evaluation import coverage_at_budget, support_metrics
from supportcover_rag.freeze import build_frozen_manifest, canonical_sha256
from supportcover_rag.generation import TokenCounter, build_token_counter
from supportcover_rag.io_utils import ensure_dir, read_csv_rows, write_csv, write_json, write_jsonl, write_yaml
from supportcover_rag.packing import SupportCoverSelector, apply_variant, build_sentence_candidates, pack_mmr
from supportcover_rag.retrieval import BM25ParagraphRetriever
from supportcover_rag.sensitivity import build_ofat_descriptors, validate_sensitivity_role
from supportcover_rag.splits import (
    load_json_ids,
    ordered_ids_sha256,
    validate_disjoint_splits,
)
from supportcover_rag.types import HotpotExample, PackedEvidence


CANONICAL_COMPONENT_VARIANTS = (
    "full",
    "no_query_coverage",
    "no_title_gain",
    "no_redundancy",
    "no_token_penalty",
)
CANONICAL_OFAT_GRIDS = {
    "beta": (0.3, 0.6, 1.2, 1.8, 2.4),
    "title": (0.0, 0.15, 0.30, 0.45, 0.60),
    "delta": (0.0, 0.075, 0.15, 0.225, 0.30),
    "gamma": (0.0, 0.15, 0.30, 0.60, 0.90),
}
CANONICAL_MMR_LAMBDAS = (0.3, 0.5, 0.7, 0.9)
DEVELOPMENT_GENERATION_COLUMNS = (
    "config_id",
    "method",
    "experiment_id",
    "alpha_relevance",
    "beta_coverage",
    "gamma_redundancy",
    "delta_token_cost",
    "title_bonus",
    "mmr_lambda_relevance",
    "num_examples",
    "answer_em",
    "answer_f1",
    "support_f1",
    "support_recall",
    "coverage_at_budget",
    "evidence_tokens",
    "split_sha256",
    "source_run_directory",
    "source_summary",
    "source_summary_sha256",
    "source_predictions",
    "source_predictions_sha256",
)
REQUIRED_EVIDENCE_ARTIFACTS = (
    "packing_screen",
    "generation_validation",
    "mmr_selection",
    "component_ablation",
)


@dataclass(frozen=True, slots=True)
class PackingPlanItem:
    study: str
    config_id: str
    supportcover: SupportCoverConfig | None = None
    factor: str = ""
    value: float | None = None
    mmr_lambda: float | None = None
    variant: str = ""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_development_protocol(config: AppConfig) -> tuple[list[str], str, Path]:
    """Fail closed unless this is the permanently frozen HotpotQA-train development protocol."""
    validate_sensitivity_role(config.split.role)
    if config.experiments.split.strip().lower() != "train":
        raise ValueError("Phase-3 tuning must use experiments.split='train'.")
    if config.runtime.limit is not None:
        raise ValueError("Phase-3 scientific runs must use all 2,000 development IDs (runtime.limit=null).")

    ids_path = Path(config.split.ids_file)
    if ids_path.name != "development_ids.json":
        raise ValueError("Phase-3 tuning must use data/splits/development_ids.json.")
    ids = load_json_ids(ids_path)
    actual_sha256 = ordered_ids_sha256(ids)
    tuning = config.development_tuning
    if len(ids) != tuning.expected_development_count:
        raise ValueError(
            f"Development count mismatch: expected {tuning.expected_development_count}, got {len(ids)}."
        )
    if actual_sha256 != tuning.expected_development_sha256:
        raise ValueError(
            "Development split SHA256 mismatch: "
            f"expected {tuning.expected_development_sha256}, got {actual_sha256}."
        )

    processed_path = Path(config.paths.data_root) / "processed" / "train.jsonl"
    if not processed_path.is_file():
        raise FileNotFoundError(f"Development population not found: {processed_path}")
    return ids, actual_sha256, processed_path


def build_packing_plan(config: AppConfig) -> list[PackingPlanItem]:
    for factor, expected in CANONICAL_OFAT_GRIDS.items():
        actual = tuple(float(value) for value in getattr(config.sensitivity, factor))
        if actual != expected:
            raise ValueError(f"Phase-3 {factor} grid must be exactly {list(expected)}; got {list(actual)}.")
    actual_mmr = tuple(float(value) for value in config.development_tuning.mmr_lambdas)
    if actual_mmr != CANONICAL_MMR_LAMBDAS:
        raise ValueError(
            f"Phase-3 MMR grid must be exactly {list(CANONICAL_MMR_LAMBDAS)}; got {list(actual_mmr)}."
        )
    plan = [
        PackingPlanItem(
            study="supportcover_sensitivity",
            config_id=f"supportcover_{descriptor.config_field}_{descriptor.value:g}",
            supportcover=descriptor.supportcover,
            factor=descriptor.config_field,
            value=descriptor.value,
        )
        for descriptor in build_ofat_descriptors(
            config.supportcover,
            config.sensitivity,
            split_role=config.split.role,
        )
    ]
    plan.extend(
        PackingPlanItem(
            study="mmr_lambda",
            config_id=f"mmr_lambda_{float(value):g}",
            mmr_lambda=float(value),
        )
        for value in config.development_tuning.mmr_lambdas
    )

    variants = tuple(config.development_tuning.component_variants)
    if variants != CANONICAL_COMPONENT_VARIANTS:
        raise ValueError(
            "Phase-3 component variants must be exactly: " + ", ".join(CANONICAL_COMPONENT_VARIANTS)
        )
    plan.extend(
        PackingPlanItem(
            study="component_ablation",
            config_id=f"supportcover_{variant}",
            supportcover=config.supportcover,
            variant=variant,
        )
        for variant in variants
    )
    return plan


def _pack(item: PackingPlanItem, candidates: list[Any], token_budget: int) -> PackedEvidence:
    if item.study == "mmr_lambda":
        assert item.mmr_lambda is not None
        return pack_mmr(candidates, token_budget=token_budget, lambda_relevance=item.mmr_lambda)
    assert item.supportcover is not None
    selector = SupportCoverSelector(item.supportcover)
    if item.study == "component_ablation":
        selector = apply_variant(selector, item.variant)
    return selector.select(candidates, token_budget=token_budget)


def _packing_row(
    example: HotpotExample,
    item: PackingPlanItem,
    packed: PackedEvidence,
    split_sha256: str,
) -> dict[str, Any]:
    if packed.used_tokens > packed.token_budget:
        raise RuntimeError(
            f"Packing plan {item.config_id} exceeded its token budget for {example.example_id}."
        )
    if len(packed.support_keys) != len(set(packed.support_keys)):
        raise RuntimeError(f"Packing plan {item.config_id} duplicated a support key for {example.example_id}.")
    support = support_metrics(packed.support_keys, example.supporting_facts)
    return {
        "evaluation_scope": "packing_only",
        "development_split_sha256": split_sha256,
        "example_id": example.example_id,
        "question_type": example.qtype,
        "difficulty": example.level,
        "study": item.study,
        "config_id": item.config_id,
        "factor": item.factor,
        "value": item.value,
        "mmr_lambda": item.mmr_lambda,
        "variant": item.variant,
        **support,
        "coverage_at_budget": coverage_at_budget(packed.support_keys, example.supporting_facts),
        "evidence_tokens": packed.used_tokens,
        "token_budget": packed.token_budget,
    }


def _aggregate_packing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["study"]), str(row["config_id"])), []).append(row)
    metrics = (
        "support_em",
        "support_precision",
        "support_recall",
        "support_f1",
        "coverage_at_budget",
        "evidence_tokens",
    )
    summary: list[dict[str, Any]] = []
    for (_, _), group in grouped.items():
        first = group[0]
        summary.append(
            {
                "evaluation_scope": "packing_only",
                "development_split_sha256": first["development_split_sha256"],
                "study": first["study"],
                "config_id": first["config_id"],
                "factor": first["factor"],
                "value": first["value"],
                "mmr_lambda": first["mmr_lambda"],
                "variant": first["variant"],
                "num_examples": len(group),
                **{metric: fmean(float(row[metric]) for row in group) for metric in metrics},
            }
        )
    return summary


def run_packing_screen(
    config: AppConfig,
    *,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Run generator-free OFAT, MMR, and clean-ablation screening on development IDs."""
    ids, split_sha256, processed_path = validate_development_protocol(config)
    examples = load_examples_by_ids(processed_path, ids)
    counter = token_counter or build_token_counter(config.generation)
    retriever = BM25ParagraphRetriever(
        k1=config.retrieval.bm25_k1,
        b=config.retrieval.bm25_b,
    )
    plan = build_packing_plan(config)
    rows: list[dict[str, Any]] = []
    token_budget = config.supportcover.token_budget
    for example in examples:
        retrieved = retriever.retrieve(example, top_k=config.retrieval.top_k_paragraphs)
        candidates = build_sentence_candidates(example.question, retrieved, counter)
        for item in plan:
            rows.append(_packing_row(example, item, _pack(item, candidates, token_budget), split_sha256))

    output_dir = Path(config.development_tuning.output_dir) / "packing"
    artifact_paths = {
        "supportcover_sensitivity": output_dir / "supportcover_sensitivity.jsonl",
        "mmr_lambda": output_dir / "mmr_lambda.jsonl",
        "component_ablation": output_dir / "component_ablation.jsonl",
        "summary": output_dir / "packing_summary.csv",
        "manifest": output_dir / "packing_manifest.json",
    }
    existing = [path for path in artifact_paths.values() if path.exists()]
    if existing and not config.runtime.overwrite:
        raise FileExistsError(
            "Phase-3 packing artifacts already exist; preserve them or set runtime.overwrite=true explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    ensure_dir(output_dir)
    for study in ("supportcover_sensitivity", "mmr_lambda", "component_ablation"):
        write_jsonl(artifact_paths[study], (row for row in rows if row["study"] == study))
    write_csv(artifact_paths["summary"], _aggregate_packing_rows(rows))

    artifact_hashes = {
        key: file_sha256(path)
        for key, path in artifact_paths.items()
        if key != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_scope": "packing_only_no_generator",
        "source": "HotpotQA train",
        "processed_path": str(processed_path),
        "ids_file": config.split.ids_file,
        "development_count": len(ids),
        "development_split_sha256": split_sha256,
        "token_budget": token_budget,
        "retrieval_depth": config.retrieval.top_k_paragraphs,
        "num_plan_items": len(plan),
        "num_metric_rows": len(rows),
        "resolved_config_sha256": canonical_sha256(asdict(config)),
        "artifacts": {
            key: {"path": str(artifact_paths[key]), "sha256": digest}
            for key, digest in artifact_hashes.items()
        },
    }
    write_json(artifact_paths["manifest"], manifest)
    return manifest


def _load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _expected_generation_configs(shortlist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    supportcover_entries = [shortlist.get("base_supportcover"), *shortlist.get("supportcover_candidates", [])]
    for entry in supportcover_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("config_id"), str):
            raise ValueError("Every retained SupportCover configuration must contain config_id.")
        coefficients = entry.get("coefficients")
        if not isinstance(coefficients, dict):
            raise ValueError(f"Shortlisted configuration {entry['config_id']} is missing coefficients.")
        expected[entry["config_id"]] = {"method": "supportcover", "coefficients": coefficients}
    for raw_lambda in shortlist.get("mmr_lambdas", []):
        value = float(raw_lambda)
        expected[f"mmr_lambda_{value:g}"] = {"method": "mmr_sentence", "mmr_lambda_relevance": value}
    return expected


def aggregate_development_generation(
    *,
    shortlist_path: str | Path,
    output_path: str | Path = "outputs/development/phase3/generation_validation.csv",
) -> list[dict[str, Any]]:
    """Curate completed development run summaries without recalculating or copying predictions."""
    shortlist = _load_json_object(shortlist_path)
    expected_sha = "0e02afdcdff360d26725abe9c197a457dcbe76c92aa54338cdc146806b9ed7c6"
    if shortlist.get("development_split_sha256") != expected_sha:
        raise ValueError("Generation aggregation requires the frozen development split SHA256.")
    expected_configs = _expected_generation_configs(shortlist)
    generation_plan = shortlist.get("generation_plan")
    if not isinstance(generation_plan, list) or not generation_plan:
        raise ValueError("Shortlist must contain a non-empty generation_plan.")

    allowed_root = Path("outputs/development/phase3/runs").resolve()
    output = Path(output_path)
    if not output.resolve().is_relative_to(Path("outputs/development/phase3").resolve()):
        raise ValueError("Development generation aggregation output must stay under outputs/development/phase3/.")

    seen_configs: set[str] = set()
    curated: list[dict[str, Any]] = []
    for item in generation_plan:
        if not isinstance(item, dict):
            raise ValueError("Every generation_plan item must be a JSON object.")
        config_id = item.get("config_id")
        if not isinstance(config_id, str) or config_id not in expected_configs:
            raise ValueError(f"Generation plan references an unretained configuration: {config_id!r}.")
        if config_id in seen_configs:
            raise ValueError(f"Generation plan duplicates configuration: {config_id}.")
        seen_configs.add(config_id)

        run_dir = Path(str(item.get("run_dir", "")))
        if not run_dir.resolve().is_relative_to(allowed_root):
            raise ValueError(f"Generation run must be under {allowed_root}: {run_dir}")
        summary_path = run_dir / "summary.csv"
        predictions_path = run_dir / "predictions.jsonl"
        config_path = run_dir / "config.resolved.yaml"
        for source in (summary_path, predictions_path, config_path):
            if not source.is_file():
                raise FileNotFoundError(f"Required generation artifact is missing: {source}")

        summary_rows = read_csv_rows(summary_path)
        if len(summary_rows) != 1:
            raise ValueError(f"Expected exactly one summary row in {summary_path}; found {len(summary_rows)}.")
        summary = summary_rows[0]
        expected = expected_configs[config_id]
        if summary.get("status") != "completed":
            raise ValueError(f"Generation run is not completed: {run_dir}")
        if summary.get("method") != expected["method"]:
            raise ValueError(f"Generation method mismatch for {config_id}.")
        if summary.get("experiment_id") != item.get("experiment_id"):
            raise ValueError(f"Experiment ID mismatch for {config_id}.")
        if summary.get("split") != "train" or summary.get("split_sha256") != expected_sha:
            raise ValueError(f"Generation run {config_id} does not use the frozen development split.")
        if int(float(summary.get("num_examples", "0"))) != 2000:
            raise ValueError(f"Generation run {config_id} does not contain all 2,000 development examples.")
        if int(float(summary.get("token_budget", "0"))) != 160:
            raise ValueError(f"Generation run {config_id} does not use token budget 160.")
        if int(float(summary.get("retrieval_depth", "0"))) != 5:
            raise ValueError(f"Generation run {config_id} does not use retrieval depth 5.")

        with config_path.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle) or {}
        supportcover = resolved.get("supportcover") or {}
        retrieval = resolved.get("retrieval") or {}
        if expected["method"] == "supportcover":
            for field, expected_value in expected["coefficients"].items():
                if float(supportcover.get(field)) != float(expected_value):
                    raise ValueError(f"Resolved {field} does not match shortlist for {config_id}.")
        elif float(retrieval.get("mmr_lambda_relevance")) != float(expected["mmr_lambda_relevance"]):
            raise ValueError(f"Resolved MMR lambda does not match shortlist for {config_id}.")

        row = {
            "config_id": config_id,
            "method": summary["method"],
            "experiment_id": summary["experiment_id"],
            "alpha_relevance": supportcover.get("alpha_relevance", ""),
            "beta_coverage": supportcover.get("beta_coverage", ""),
            "gamma_redundancy": supportcover.get("gamma_redundancy", ""),
            "delta_token_cost": supportcover.get("delta_token_cost", ""),
            "title_bonus": supportcover.get("title_bonus", ""),
            "mmr_lambda_relevance": retrieval.get("mmr_lambda_relevance", ""),
            "num_examples": summary["num_examples"],
            "answer_em": summary["answer_em"],
            "answer_f1": summary["answer_f1"],
            "support_f1": summary["support_f1"],
            "support_recall": summary["support_recall"],
            "coverage_at_budget": summary["coverage_at_budget"],
            "evidence_tokens": summary["evidence_tokens"],
            "split_sha256": summary["split_sha256"],
            "source_run_directory": str(run_dir),
            "source_summary": str(summary_path),
            "source_summary_sha256": file_sha256(summary_path),
            "source_predictions": str(predictions_path),
            "source_predictions_sha256": file_sha256(predictions_path),
        }
        curated.append({column: row[column] for column in DEVELOPMENT_GENERATION_COLUMNS})

    if seen_configs != set(expected_configs):
        missing = sorted(set(expected_configs) - seen_configs)
        raise ValueError("Generation plan is missing retained configurations: " + ", ".join(missing))
    write_csv(output, curated)
    return curated


def _read_phase3_decision(path: str | Path, config: AppConfig) -> dict[str, Any]:
    build_packing_plan(config)
    decision_path = Path(path)
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    if not isinstance(decision, dict):
        raise ValueError("The Phase-3 decision record must be a JSON object.")
    if decision.get("development_split_sha256") != config.development_tuning.expected_development_sha256:
        raise ValueError("The decision record does not identify the frozen development split SHA256.")

    coefficients = decision.get("selected_supportcover_coefficients")
    required = {
        "alpha_relevance",
        "beta_coverage",
        "gamma_redundancy",
        "delta_token_cost",
        "title_bonus",
    }
    if not isinstance(coefficients, dict) or set(coefficients) != required:
        raise ValueError("The decision record must contain exactly the five selected SupportCover coefficients.")
    if float(coefficients["alpha_relevance"]) != config.supportcover.alpha_relevance:
        raise ValueError("alpha_relevance was not a Phase-3 tuning factor and must remain fixed.")
    grid_by_field = {
        "beta_coverage": config.sensitivity.beta,
        "title_bonus": config.sensitivity.title,
        "delta_token_cost": config.sensitivity.delta,
        "gamma_redundancy": config.sensitivity.gamma,
    }
    for field, grid in grid_by_field.items():
        if float(coefficients[field]) not in [float(value) for value in grid]:
            raise ValueError(f"Selected {field}={coefficients[field]} is outside the pre-registered OFAT grid.")

    mmr_lambda = decision.get("selected_mmr_lambda_relevance")
    if not isinstance(mmr_lambda, (int, float)) or float(mmr_lambda) not in [
        float(value) for value in config.development_tuning.mmr_lambdas
    ]:
        raise ValueError("The selected MMR lambda is outside the pre-registered development grid.")

    evidence = decision.get("evidence_artifacts")
    if not isinstance(evidence, dict):
        raise ValueError("The decision record must contain evidence_artifacts.")
    missing = [name for name in REQUIRED_EVIDENCE_ARTIFACTS if not evidence.get(name)]
    if missing:
        raise ValueError("The decision record is missing required evidence artifacts: " + ", ".join(missing))
    development_output = Path(config.development_tuning.output_dir).resolve()
    for name in REQUIRED_EVIDENCE_ARTIFACTS:
        evidence_path = Path(evidence[name]).resolve()
        if not evidence_path.is_relative_to(development_output):
            raise ValueError(
                f"Evidence artifact '{name}' must be inside the development-only output directory: "
                f"{development_output}"
            )
    return decision


def freeze_development_selection(
    config: AppConfig,
    *,
    decision_path: str | Path,
    final_ids_path: str | Path,
    final_config_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Freeze development-selected settings without loading final examples or predictions."""
    development_ids, development_sha256, _ = validate_development_protocol(config)
    final_ids = load_json_ids(final_ids_path)
    final_sha256 = ordered_ids_sha256(final_ids)
    tuning = config.development_tuning
    if len(final_ids) != tuning.expected_final_count or final_sha256 != tuning.expected_final_sha256:
        raise ValueError("Final split count or SHA256 does not match the permanently frozen protocol.")
    validate_disjoint_splits({"development": development_ids, "final": final_ids})

    decision = _read_phase3_decision(decision_path, config)
    evidence_hashes: dict[str, Any] = {}
    for name in REQUIRED_EVIDENCE_ARTIFACTS:
        evidence_path = Path(decision["evidence_artifacts"][name])
        if not evidence_path.is_file():
            raise FileNotFoundError(f"Required development evidence artifact not found: {evidence_path}")
        evidence_hashes[name] = {"path": str(evidence_path), "sha256": file_sha256(evidence_path)}
    evidence_hashes["decision_record"] = {
        "path": str(decision_path),
        "sha256": file_sha256(decision_path),
    }

    coefficients = {key: float(value) for key, value in decision["selected_supportcover_coefficients"].items()}
    selected_supportcover = replace(config.supportcover, **coefficients)
    selected_mmr = float(decision["selected_mmr_lambda_relevance"])
    manifest = build_frozen_manifest(
        supportcover_coefficients=coefficients,
        mmr_lambda_relevance=selected_mmr,
        development_split_sha256=development_sha256,
        final_split_sha256=final_sha256,
        dataset={
            "path": config.raw_data.dataset_path,
            "config": config.raw_data.dataset_config,
            "final_source": "HotpotQA validation complete population",
            "final_count": len(final_ids),
        },
        model=asdict(config.generation),
        prompt_settings=asdict(config.prompting),
        decoding_settings={
            "temperature": config.generation.temperature,
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
        },
        token_budget=selected_supportcover.token_budget,
        retrieval_depth=config.retrieval.top_k_paragraphs,
        selection_evidence=evidence_hashes,
    )
    manifest.update(
        {
            "selection_role": "development_only",
            "final_predictions_inspected": False,
            "final_manifest_only_access": True,
        }
    )

    frozen_config = asdict(
        replace(
            config,
            runtime=replace(config.runtime, limit=None, overwrite=False, resume=True),
            raw_data=replace(config.raw_data, splits=["validation"]),
            freeze=replace(
                config.freeze,
                manifest_file=str(manifest_path),
                sha256=manifest["config_sha256"],
                require_sha256=True,
            ),
            split=replace(
                config.split,
                ids_file=str(final_ids_path),
                role="final",
                stratify_by=[],
            ),
            retrieval=replace(config.retrieval, mmr_lambda_relevance=selected_mmr),
            supportcover=selected_supportcover,
            experiments=replace(config.experiments, split="validation"),
        )
    )
    write_yaml(final_config_path, frozen_config)
    write_json(manifest_path, manifest)
    return manifest
