from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from supportcover_rag.freeze import canonical_json, canonical_sha256


ROBUSTNESS_FAMILIES = ("budget", "models", "cross_dataset")
PRIMARY_FINAL_ROLE = "primary_final"
CROSS_DATASET_ROLE = "cross_dataset_robustness"
SUPPORTCOVER_COEFFICIENTS = (
    "alpha_relevance",
    "beta_coverage",
    "gamma_redundancy",
    "delta_token_cost",
    "title_bonus",
)
AGGREGATE_METRICS = (
    "answer_em",
    "answer_f1",
    "support_precision",
    "support_recall",
    "support_f1",
    "coverage_at_budget",
    "evidence_tokens",
    "retrieval_latency_ms",
    "packing_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "peak_rss_mb",
)
UNRESOLVED_MARKERS = {"", "UNRESOLVED", "__FROM_PHASE3_FREEZE__", "__PREREGISTER_LATER__"}


@dataclass(frozen=True, slots=True)
class RobustnessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RobustnessReadinessReport:
    checks: tuple[RobustnessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_scope": "metadata_only_no_final_outcomes",
            "checks": [asdict(check) for check in self.checks],
            "robustness_execution": "READY" if self.ready else "BLOCKED",
        }


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    name: str
    config_or_version: str
    split: str
    source_revision: str | None
    license_or_reference: str | None


@dataclass(frozen=True, slots=True)
class CanonicalMultiHopExample:
    example_id: str
    question: str
    answer: str
    context: tuple[dict[str, Any], ...]
    supporting_facts: tuple[tuple[str, int], ...] | None
    provenance: DatasetProvenance


class MultiHopQAAdapter(Protocol):
    @property
    def provenance(self) -> DatasetProvenance: ...

    def adapt(self, record: Mapping[str, Any]) -> CanonicalMultiHopExample: ...


@dataclass(frozen=True, slots=True)
class FrozenPackedEvidence:
    example_id: str
    retrieved_source_ids: tuple[str, ...]
    selected_support_keys: tuple[tuple[str, int], ...] | None
    rendered_evidence: str
    used_evidence_tokens: int
    token_budget: int
    method: str
    config_id: str
    freeze_sha256: str
    split_sha256: str
    retrieval_depth: int
    retrieval_settings: Mapping[str, Any]
    packing_tokenizer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_frozen_packed_evidence(
    path: str | Path, records: Sequence[FrozenPackedEvidence]
) -> None:
    if not records:
        raise ValueError("Frozen packed-evidence serialization requires at least one record.")
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen packed evidence contains duplicate example IDs.")
    protocol_fields = (
        "freeze_sha256",
        "split_sha256",
        "method",
        "config_id",
        "token_budget",
        "retrieval_depth",
        "retrieval_settings",
        "packing_tokenizer",
    )
    reference = records[0]
    for record in records[1:]:
        changed = [
            field for field in protocol_fields
            if canonical_json(getattr(record, field)) != canonical_json(getattr(reference, field))
        ]
        if changed:
            raise ValueError("Packed-evidence records mix protocols: " + ", ".join(changed))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record.to_dict()) + "\n")


def load_frozen_packed_evidence(path: str | Path) -> list[FrozenPackedEvidence]:
    records: list[FrozenPackedEvidence] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Packed-evidence line {line_number} must be an object.")
            support_keys = value.get("selected_support_keys")
            records.append(
                FrozenPackedEvidence(
                    example_id=str(value["example_id"]),
                    retrieved_source_ids=tuple(value["retrieved_source_ids"]),
                    selected_support_keys=(
                        None if support_keys is None else tuple((str(key[0]), int(key[1])) for key in support_keys)
                    ),
                    rendered_evidence=str(value["rendered_evidence"]),
                    used_evidence_tokens=int(value["used_evidence_tokens"]),
                    token_budget=int(value["token_budget"]),
                    method=str(value["method"]),
                    config_id=str(value["config_id"]),
                    freeze_sha256=_require_sha(value["freeze_sha256"], "packed-evidence freeze SHA256"),
                    split_sha256=_require_sha(value["split_sha256"], "packed-evidence split SHA256"),
                    retrieval_depth=int(value["retrieval_depth"]),
                    retrieval_settings=dict(value["retrieval_settings"]),
                    packing_tokenizer=str(value["packing_tokenizer"]),
                )
            )
    if not records:
        raise ValueError("Frozen packed-evidence artifact is empty.")
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen packed-evidence artifact contains duplicate example IDs.")
    return records


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {label}: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _load_yaml(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {label}: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256 digest.")
    return value.lower()


def _require_resolved(value: object, label: str) -> object:
    if value is None or (isinstance(value, str) and value.strip() in UNRESOLVED_MARKERS):
        raise ValueError(f"{label} remains unresolved.")
    return value


def validate_freeze_boundary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Phase-6 identity without reading examples, predictions, or outcomes."""
    if manifest.get("schema_version") != 2:
        raise ValueError("Phase-3 freeze manifest schema_version must be 2.")
    if manifest.get("selection_role") != "development_only":
        raise ValueError("Phase-3 freeze must record development-only selection.")
    if manifest.get("final_predictions_inspected") is not False:
        raise ValueError("Phase-3 freeze must record final_predictions_inspected=false.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Phase-3 freeze is missing configuration.")
    freeze_sha = _require_sha(manifest.get("config_sha256"), "freeze SHA256")
    if canonical_sha256(configuration) != freeze_sha:
        raise ValueError("Phase-3 freeze SHA256 does not match its canonical configuration.")
    method = _require_resolved(configuration.get("supportcover_method"), "frozen SupportCover method")
    variant = _require_resolved(configuration.get("supportcover_variant"), "frozen SupportCover variant")
    coefficients = configuration.get("supportcover_coefficients")
    if not isinstance(coefficients, Mapping) or set(coefficients) != set(SUPPORTCOVER_COEFFICIENTS):
        raise ValueError("Freeze must contain exactly the five SupportCover coefficients.")
    for name in SUPPORTCOVER_COEFFICIENTS:
        value = coefficients[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Frozen {name} must be finite numeric data.")
    token_budget = configuration.get("token_budget")
    retrieval_depth = configuration.get("retrieval_depth")
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget <= 0:
        raise ValueError("Frozen token budget must be a positive integer.")
    if not isinstance(retrieval_depth, int) or isinstance(retrieval_depth, bool) or retrieval_depth <= 0:
        raise ValueError("Frozen retrieval depth must be a positive integer.")
    model = configuration.get("model")
    if not isinstance(model, Mapping) or not model.get("model_name_or_path"):
        raise ValueError("Freeze is missing the primary generator identity.")
    _require_sha(configuration.get("final_split_sha256"), "frozen final split SHA256")
    return {
        "freeze_sha256": freeze_sha,
        "method": str(method),
        "variant": str(variant),
        "coefficients": dict(coefficients),
        "token_budget": token_budget,
        "retrieval_depth": retrieval_depth,
        "mmr_lambda_relevance": configuration.get("mmr_lambda_relevance"),
        "primary_generator": dict(model),
        "prompt_settings": configuration.get("prompt_settings"),
        "decoding_settings": configuration.get("decoding_settings"),
        "dataset": configuration.get("dataset"),
        "final_split_sha256": configuration["final_split_sha256"],
    }


def validate_main_study_completion(
    completion: Mapping[str, Any], *, freeze_sha256: str, split_sha256: str
) -> None:
    if completion.get("schema_version") != 1 or completion.get("status") != "completed":
        raise ValueError("Phase-5 main-study completion manifest is missing or incomplete.")
    if completion.get("scientific_role") != PRIMARY_FINAL_ROLE:
        raise ValueError("Phase-5 completion must identify the primary_final role.")
    if completion.get("freeze_sha256") != freeze_sha256:
        raise ValueError("Phase-5 completion freeze SHA256 does not match Phase 3.")
    if completion.get("split_sha256") != split_sha256:
        raise ValueError("Phase-5 completion split SHA256 does not match the primary final population.")
    count = completion.get("num_examples")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Phase-5 completion must record a positive number of examples.")
    run_ids = completion.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("Phase-5 completion must record unique non-empty run IDs.")
    result = completion.get("main_results")
    if not isinstance(result, Mapping) or not result.get("path"):
        raise ValueError("Phase-5 completion must identify the main-results artifact.")
    result_path = Path(str(result["path"]))
    expected_sha = _require_sha(result.get("sha256"), "Phase-5 main-results SHA256")
    if not result_path.is_file() or _sha256_file(result_path) != expected_sha:
        raise ValueError("Phase-5 main-results artifact is missing or has the wrong SHA256.")


def build_main_study_completion(
    *,
    main_results_path: str | Path,
    freeze_sha256: str,
    split_sha256: str,
    num_examples: int,
    run_ids: Sequence[str],
    code_revision: str | None,
    created_at: str,
) -> dict[str, Any]:
    """Build completion provenance only after the real Phase-5 aggregate exists."""
    results = Path(main_results_path)
    if not results.is_file():
        raise FileNotFoundError(f"Missing completed Phase-5 main-results artifact: {results}")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "scientific_role": PRIMARY_FINAL_ROLE,
        "freeze_sha256": _require_sha(freeze_sha256, "freeze SHA256"),
        "split_sha256": _require_sha(split_sha256, "final split SHA256"),
        "num_examples": num_examples,
        "run_ids": list(run_ids),
        "main_results": {"path": str(results), "sha256": _sha256_file(results)},
        "code_revision": code_revision,
        "created_at": created_at,
    }
    validate_main_study_completion(
        manifest,
        freeze_sha256=str(manifest["freeze_sha256"]),
        split_sha256=str(manifest["split_sha256"]),
    )
    return manifest


def frozen_protocol_descriptor(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": identity["method"],
        "variant": identity["variant"],
        "coefficients": identity["coefficients"],
        "retrieval_depth": identity["retrieval_depth"],
        "mmr_lambda_relevance": identity["mmr_lambda_relevance"],
        "token_budget": identity["token_budget"],
        "generator": identity["primary_generator"],
        "prompt_settings": identity["prompt_settings"],
        "decoding_settings": identity["decoding_settings"],
        "dataset": identity["dataset"],
        "split_role": PRIMARY_FINAL_ROLE,
        "split_sha256": identity["final_split_sha256"],
    }


def _validate_only_allowed_changes(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, allowed: set[str], family: str
) -> None:
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    if missing or extra:
        raise ValueError(f"{family} protocol fields differ: missing={missing}, extra={extra}.")
    changed = {key for key in reference if canonical_json(reference[key]) != canonical_json(candidate[key])}
    forbidden = sorted(changed - allowed)
    if forbidden:
        raise ValueError(f"{family} changed forbidden scientific fields: {', '.join(forbidden)}")


def validate_budget_isolation(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    _validate_only_allowed_changes(reference, candidate, allowed={"token_budget"}, family="budget robustness")
    budget = candidate.get("token_budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("Budget robustness token_budget must be a positive integer.")


def validate_model_isolation(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    _validate_only_allowed_changes(reference, candidate, allowed={"generator"}, family="model robustness")
    generator = candidate.get("generator")
    required = ("implementation", "model_id", "backend", "precision", "prompt_format_id", "batch_size")
    if not isinstance(generator, Mapping) or any(not generator.get(key) for key in required):
        raise ValueError("Model robustness generator descriptor is incomplete.")


def validate_cross_dataset_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("role") != CROSS_DATASET_ROLE:
        raise ValueError("Cross-dataset execution requires role='cross_dataset_robustness'.")
    dataset = protocol.get("dataset")
    required = ("name", "version_or_revision", "split", "id_file", "count", "id_sha256")
    if not isinstance(dataset, Mapping) or any(_is_unresolved(dataset.get(key)) for key in required):
        raise ValueError("Cross-dataset provenance is incomplete.")
    if protocol.get("tuning_permitted") is not False:
        raise ValueError("Cross-dataset results must never permit tuning.")
    if protocol.get("primary_final_split_sha256") == dataset.get("id_sha256"):
        raise ValueError("Primary-final population cannot masquerade as cross-dataset robustness.")


def _is_unresolved(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in UNRESOLVED_MARKERS)


def validate_plan(plan: Mapping[str, Any], family: str, identity: Mapping[str, Any] | None = None) -> None:
    if plan.get("schema_version") != 1 or plan.get("robustness_family") != family:
        raise ValueError(f"Expected a schema-1 {family} robustness plan.")
    if plan.get("freeze_manifest") != "configs/frozen/final_manifest.json":
        raise ValueError("Robustness plan must use the canonical Phase-3 freeze manifest.")
    if plan.get("main_study_completion_manifest") != "outputs/final/main_study_completion.json":
        raise ValueError("Robustness plan must use the canonical Phase-5 completion manifest.")
    if identity is not None and plan.get("freeze_sha256") != identity["freeze_sha256"]:
        raise ValueError("Robustness plan freeze SHA256 is unresolved or mismatched.")
    if identity is not None and family in {"budget", "models"}:
        population = plan.get("population")
        final_count = identity["dataset"].get("final_count") if isinstance(identity.get("dataset"), Mapping) else None
        if not isinstance(population, Mapping) or population.get("role") != PRIMARY_FINAL_ROLE:
            raise ValueError("Robustness plan must identify the primary_final population.")
        if population.get("ids_file") != "data/splits/final_ids.json":
            raise ValueError("Robustness plan must use the canonical final ID manifest.")
        if population.get("split_sha256") != identity["final_split_sha256"] or population.get("count") != final_count:
            raise ValueError("Robustness plan final population does not match the Phase-3 freeze.")
        if canonical_json(plan.get("frozen_protocol")) != canonical_json(frozen_protocol_descriptor(identity)):
            raise ValueError("Robustness plan frozen method/protocol does not match the Phase-3 freeze.")
    if family == "budget":
        budgets = plan.get("supported_budgets")
        if not isinstance(budgets, list) or not budgets or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets
        ) or len(budgets) != len(set(budgets)):
            raise ValueError("Budget plan requires unique positive ordered supported_budgets.")
        if plan.get("allowed_changed_fields") != ["token_budget"]:
            raise ValueError("Budget plan may change only token_budget.")
    elif family == "models":
        if plan.get("allowed_changed_fields") != ["generator"]:
            raise ValueError("Model plan may change only generator identity.")
        if not isinstance(plan.get("models"), list):
            raise ValueError("Model plan models must be an explicit list.")
    elif family == "cross_dataset":
        validate_cross_dataset_protocol(plan)


def validate_packed_evidence_reuse(
    cached: FrozenPackedEvidence, requested: Mapping[str, Any]
) -> None:
    comparisons = {
        "freeze_sha256": cached.freeze_sha256,
        "split_sha256": cached.split_sha256,
        "method": cached.method,
        "config_id": cached.config_id,
        "token_budget": cached.token_budget,
        "retrieval_depth": cached.retrieval_depth,
        "retrieval_settings": cached.retrieval_settings,
        "packing_tokenizer": cached.packing_tokenizer,
    }
    mismatches = [
        key for key, value in comparisons.items()
        if key not in requested or canonical_json(requested[key]) != canonical_json(value)
    ]
    if mismatches:
        raise ValueError("Frozen packed evidence is not reusable; changed: " + ", ".join(mismatches))


def deterministic_run_id(family: str, variation: Mapping[str, Any], freeze_sha256: str) -> str:
    if family not in ROBUSTNESS_FAMILIES:
        raise ValueError(f"Unknown robustness family: {family}")
    suffix = canonical_sha256({"family": family, "variation": variation, "freeze": freeze_sha256})[:12]
    return f"phase6-{family}-{suffix}"


def _read_run(run_dir: str | Path, family: str) -> dict[str, Any]:
    source = Path(run_dir)
    metrics = _load_json(source / "metrics.json", "robustness metrics")
    config = _load_yaml(source / "config.resolved.yaml", "resolved robustness config")
    provenance = config.get("robustness_run")
    if not isinstance(provenance, Mapping) or provenance.get("family") != family:
        raise ValueError(f"Run {source} does not identify the {family} robustness family.")
    if metrics.get("status") != "completed":
        raise ValueError(f"Robustness run is incomplete: {source}")
    if metrics.get("num_examples") != provenance.get("num_examples"):
        raise ValueError(f"Robustness run example count mismatch: {source}")
    expected_role = CROSS_DATASET_ROLE if family == "cross_dataset" else PRIMARY_FINAL_ROLE
    if provenance.get("scientific_role") != expected_role:
        raise ValueError(f"Robustness run has the wrong scientific role: {source}")
    for field in ("freeze_sha256", "split_sha256", "config_sha256"):
        _require_sha(provenance.get(field), f"run {field}")
    predictions_path = source / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing robustness predictions: {predictions_path}")
    example_ids: list[str] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            prediction = json.loads(line)
            example_id = prediction.get("example_id") if isinstance(prediction, Mapping) else None
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"Prediction line {line_number} has no example_id: {predictions_path}")
            example_ids.append(example_id)
    if len(example_ids) != metrics["num_examples"] or len(example_ids) != len(set(example_ids)):
        raise ValueError(f"Robustness prediction population is incomplete or duplicated: {source}")
    row: dict[str, Any] = {}
    if family == "budget":
        row["budget"] = provenance.get("token_budget")
    elif family == "models":
        model = provenance.get("generator") or {}
        row.update({
            "model": model.get("model_id"),
            "model_revision": model.get("revision"),
            "backend": model.get("backend"),
            "precision": model.get("precision"),
            "batch_size": model.get("batch_size"),
            "packing_tokenizer": provenance.get("packing_tokenizer"),
            "generation_tokenizer": model.get("generation_tokenizer"),
            "prompt_token_count": metrics.get("prompt_token_count"),
            "peak_memory_mb": metrics.get("peak_memory_mb"),
        })
    else:
        dataset = provenance.get("dataset") or {}
        row.update({
            "dataset": dataset.get("name"),
            "dataset_version": dataset.get("version_or_revision"),
            "dataset_split": dataset.get("split"),
            "dataset_id_sha256": dataset.get("id_sha256"),
        })
    row["N"] = metrics.get("num_examples")
    row.update({metric: metrics.get(metric) for metric in AGGREGATE_METRICS})
    row.update({
        "split_sha256": provenance.get("split_sha256"),
        "freeze_sha256": provenance.get("freeze_sha256"),
        "config_sha256": provenance.get("config_sha256"),
        "code_revision": provenance.get("code_revision"),
        "run_id": provenance.get("run_id"),
        "run_dir": str(source),
    })
    for name, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Robustness aggregate field {name} contains NaN or infinity.")
    return row


def aggregate_robustness_runs(
    family: str, run_dirs: Sequence[str | Path], *, output_path: str | Path
) -> list[dict[str, Any]]:
    if family not in ROBUSTNESS_FAMILIES:
        raise ValueError(f"Unknown robustness family: {family}")
    if not run_dirs:
        raise ValueError("At least one completed robustness run is required.")
    rows = [_read_run(run_dir, family) for run_dir in run_dirs]
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Robustness aggregation found duplicate run IDs.")
    for field in ("split_sha256", "freeze_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"Robustness aggregation found mismatched {field} values.")
    rows.sort(key=lambda row: (str(row.get("budget", "")), str(row.get("model", "")), str(row.get("dataset", ""))))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_robustness_manifest(
    *,
    family: str,
    freeze_sha256: str,
    main_study_completion: Mapping[str, Any],
    run_dirs: Sequence[str | Path],
    aggregate_path: str | Path,
    population: Mapping[str, Any],
    frozen_method: Mapping[str, Any],
    allowed_changed_fields: Sequence[str],
    forbidden_changed_fields: Sequence[str],
    code_revision: str | None,
    environment_reference: str | None,
    created_at: str,
) -> dict[str, Any]:
    if not run_dirs or not Path(aggregate_path).is_file():
        raise ValueError("A completed aggregate and at least one source run are required.")
    sources = []
    for run_dir in run_dirs:
        source = Path(run_dir)
        sources.append({
            "run_id": source.name,
            "path": str(source),
            "metrics_sha256": _sha256_file(source / "metrics.json"),
            "predictions_sha256": _sha256_file(source / "predictions.jsonl"),
            "config_sha256": _sha256_file(source / "config.resolved.yaml"),
        })
    return {
        "schema_version": 1,
        "status": "completed",
        "phase3_freeze_sha256": _require_sha(freeze_sha256, "freeze SHA256"),
        "main_study_completion": dict(main_study_completion),
        "robustness_family": family,
        "source_artifacts": sources,
        "population": dict(population),
        "frozen_method": dict(frozen_method),
        "allowed_changed_fields": list(allowed_changed_fields),
        "forbidden_changed_fields": list(forbidden_changed_fields),
        "code_revision": code_revision,
        "environment_reference": environment_reference,
        "created_at": created_at,
        "aggregate": {"path": str(aggregate_path), "sha256": _sha256_file(aggregate_path)},
    }


def verify_robustness_readiness(
    *,
    freeze_manifest_path: str | Path,
    main_completion_path: str | Path,
    budget_plan_path: str | Path,
    model_plan_path: str | Path,
    cross_dataset_plan_path: str | Path,
    output_path: str | Path | None = None,
) -> RobustnessReadinessReport:
    checks: list[RobustnessCheck] = []
    identity: dict[str, Any] | None = None
    try:
        identity = validate_freeze_boundary(_load_json(freeze_manifest_path, "Phase-3 freeze manifest"))
        checks.append(RobustnessCheck("Phase-3 freeze", True, "validated canonical frozen method identity"))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        checks.append(RobustnessCheck("Phase-3 freeze", False, str(exc)))

    try:
        if identity is None:
            raise ValueError("blocked until Phase-3 freeze passes")
        completion = _load_json(main_completion_path, "Phase-5 main-study completion manifest")
        validate_main_study_completion(
            completion,
            freeze_sha256=identity["freeze_sha256"],
            split_sha256=identity["final_split_sha256"],
        )
        checks.append(RobustnessCheck("Phase-5 main study completion", True, "validated completion provenance"))
    except (FileNotFoundError, ValueError) as exc:
        checks.append(RobustnessCheck("Phase-5 main study completion", False, str(exc)))

    for name, family, path in (
        ("budget robustness plan", "budget", budget_plan_path),
        ("model robustness plan", "models", model_plan_path),
        ("cross-dataset protocol", "cross_dataset", cross_dataset_plan_path),
    ):
        try:
            plan = _load_yaml(path, name)
            validate_plan(plan, family, identity)
            checks.append(RobustnessCheck(name, True, "validated"))
        except (FileNotFoundError, ValueError, TypeError) as exc:
            checks.append(RobustnessCheck(name, False, str(exc)))

    if identity is None:
        checks.append(RobustnessCheck("frozen-method integrity", False, "blocked until Phase-3 freeze passes"))
    else:
        checks.append(RobustnessCheck("frozen-method integrity", True, "coefficients and method derive from freeze"))
    report = RobustnessReadinessReport(tuple(checks))
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report
