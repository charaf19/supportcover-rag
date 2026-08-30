from __future__ import annotations

import importlib.util
import importlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from supportcover_rag.config import AppConfig, ExternalCompressorConfig, load_config
from supportcover_rag.freeze import canonical_sha256
from supportcover_rag.io_utils import read_jsonl, write_csv, write_json, write_yaml
from supportcover_rag.splits import load_json_ids, ordered_ids_sha256, validate_disjoint_splits, validate_unique_ids


PRIMARY_MAIN_METHODS = (
    "paragraph_topk",
    "relevance_only",
    "mmr_sentence",
    "greedy_query_cover",
    "supportcover_final",
)
OPTIONAL_EXTERNAL_METHOD = "external_compressor"
FINAL_MANIFEST_SCHEMA_VERSION = 2
_SUPPORTCOVER_FIELDS = (
    "alpha_relevance",
    "beta_coverage",
    "gamma_redundancy",
    "delta_token_cost",
    "title_bonus",
)
_RESULT_FIELDS = (
    "answer_em",
    "answer_f1",
    "support_em",
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


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class FinalReadinessReport:
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_scope": "metadata_only_no_final_outcomes",
            "checks": [asdict(check) for check in self.checks],
            "final_main_study": "READY" if self.ready else "BLOCKED",
        }


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {label}: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256 digest.")
    return value.lower()


def _require_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite numeric value.")
    return float(value)


def validate_frozen_manifest(manifest: Mapping[str, Any], config: AppConfig) -> Mapping[str, Any]:
    if manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Freeze manifest schema_version must be {FINAL_MANIFEST_SCHEMA_VERSION}.")
    if manifest.get("selection_role") != "development_only":
        raise ValueError("Freeze manifest must record selection_role='development_only'.")
    if manifest.get("final_predictions_inspected") is not False:
        raise ValueError("Freeze manifest must record final_predictions_inspected=false.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Freeze manifest is missing its configuration object.")
    manifest_sha = _require_sha(manifest.get("config_sha256"), label="freeze config_sha256")
    if canonical_sha256(configuration) != manifest_sha:
        raise ValueError("Freeze manifest config_sha256 does not match its canonical configuration.")
    development_sha = _require_sha(
        configuration.get("development_split_sha256"),
        label="frozen development split SHA256",
    )
    final_sha = _require_sha(configuration.get("final_split_sha256"), label="frozen final split SHA256")
    if development_sha != config.final_study.expected_development_sha256:
        raise ValueError("Freeze manifest development split SHA256 does not match the frozen protocol.")
    if final_sha != config.final_study.expected_final_sha256:
        raise ValueError("Freeze manifest final split SHA256 does not match the frozen protocol.")
    dataset = configuration.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("final_count") != config.final_study.expected_final_count:
        raise ValueError("Freeze manifest final population count does not match the frozen protocol.")
    coefficients = configuration.get("supportcover_coefficients")
    if not isinstance(coefficients, Mapping):
        raise ValueError("Freeze manifest is missing SupportCover coefficients.")
    for field in _SUPPORTCOVER_FIELDS:
        _require_number(coefficients.get(field), label=f"frozen {field}")
    mmr_lambda = _require_number(configuration.get("mmr_lambda_relevance"), label="frozen MMR lambda")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("Frozen MMR lambda must be between 0 and 1.")
    token_budget = configuration.get("token_budget")
    retrieval_depth = configuration.get("retrieval_depth")
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget <= 0:
        raise ValueError("Frozen token budget must be a positive integer.")
    if not isinstance(retrieval_depth, int) or isinstance(retrieval_depth, bool) or retrieval_depth <= 0:
        raise ValueError("Frozen retrieval depth must be a positive integer.")
    return configuration


def validate_external_compressor_configuration(config: ExternalCompressorConfig) -> None:
    if not config.enabled:
        raise ValueError("No external compressor implementation is enabled.")
    required = {
        "adapter": config.adapter,
        "implementation_id": config.implementation_id,
        "version": config.version,
        "revision": config.revision,
    }
    missing = [field for field, value in required.items() if not value.strip()]
    if missing:
        raise ValueError("External compressor provenance is missing: " + ", ".join(missing))
    if ":" not in config.adapter:
        raise ValueError("External compressor adapter must use 'module:factory' syntax.")
    module_name, factory_name = config.adapter.split(":", maxsplit=1)
    if importlib.util.find_spec(module_name) is None:
        raise ValueError(f"External compressor module is unavailable: {module_name}")
    module = importlib.import_module(module_name)
    if not callable(getattr(module, factory_name, None)):
        raise ValueError(f"External compressor factory is unavailable: {config.adapter}")


def validate_final_execution_config(
    config: AppConfig,
    *,
    manifest_path: str | Path | None = None,
    development_ids_path: str | Path | None = None,
    final_ids_path: str | Path | None = None,
    require_external_adapter: bool = True,
    require_main_methods: bool = True,
) -> dict[str, Any]:
    """Fail closed using only config, manifest, and frozen ID identity metadata."""
    if config.split.role.strip().lower() != "final":
        raise ValueError("Final execution requires split.role='final'.")
    if config.runtime.limit is not None:
        raise ValueError("Final execution requires runtime.limit=null.")
    if config.runtime.overwrite or not config.runtime.resume:
        raise ValueError("Final execution requires runtime.overwrite=false and runtime.resume=true.")
    if config.experiments.split.strip().lower() not in {"validation", "val"}:
        raise ValueError("Final execution must use the HotpotQA validation split.")
    if [split.strip().lower() for split in config.raw_data.splits] != ["validation"]:
        raise ValueError("Final execution raw_data.splits must contain only validation.")
    manifest_source = Path(manifest_path or config.freeze.manifest_file)
    manifest = _load_json_object(manifest_source, label="Phase-3 freeze manifest")
    frozen = validate_frozen_manifest(manifest, config)
    manifest_sha = _require_sha(manifest.get("config_sha256"), label="freeze config_sha256")
    if not config.freeze.require_sha256 or config.freeze.sha256 != manifest_sha:
        raise ValueError("Final configuration freeze SHA does not match the Phase-3 manifest.")

    development_source = Path(development_ids_path or config.final_study.development_ids_file)
    final_source = Path(final_ids_path or config.final_study.final_ids_file or config.split.ids_file)
    if Path(config.split.ids_file).resolve() != final_source.resolve():
        raise ValueError("split.ids_file does not match the validated final ID manifest path.")
    development_ids = load_json_ids(development_source)
    final_ids = load_json_ids(final_source)
    validate_unique_ids(development_ids)
    validate_unique_ids(final_ids)
    if len(development_ids) != config.final_study.expected_development_count:
        raise ValueError("Development ID count does not match the frozen protocol.")
    if len(final_ids) != config.final_study.expected_final_count:
        raise ValueError("Final ID count does not match the frozen protocol.")
    development_sha = ordered_ids_sha256(development_ids)
    final_sha = ordered_ids_sha256(final_ids)
    if development_sha != config.final_study.expected_development_sha256:
        raise ValueError("Development ID SHA256 does not match the frozen protocol.")
    if final_sha != config.final_study.expected_final_sha256:
        raise ValueError("Final ID SHA256 does not match the frozen protocol.")
    validate_disjoint_splits({"development": development_ids, "final": final_ids})
    if frozen.get("development_split_sha256") != development_sha or frozen.get("final_split_sha256") != final_sha:
        raise ValueError("Freeze manifest split identities do not match the ID manifests.")

    coefficients = frozen["supportcover_coefficients"]
    for field in _SUPPORTCOVER_FIELDS:
        actual = getattr(config.supportcover, field)
        if actual is None or float(actual) != float(coefficients[field]):
            raise ValueError(f"Final configuration does not match frozen SupportCover field '{field}'.")
    if config.retrieval.mmr_lambda_relevance is None or float(config.retrieval.mmr_lambda_relevance) != float(
        frozen["mmr_lambda_relevance"]
    ):
        raise ValueError("Final configuration does not match the frozen MMR lambda.")
    if config.retrieval.top_k_paragraphs != frozen["retrieval_depth"]:
        raise ValueError("Final configuration does not match the frozen retrieval depth.")
    if config.supportcover.token_budget != frozen["token_budget"]:
        raise ValueError("Final configuration does not match the frozen token budget.")
    dataset = frozen["dataset"]
    if config.raw_data.dataset_path != dataset.get("path") or config.raw_data.dataset_config != dataset.get("config"):
        raise ValueError("Final configuration dataset does not match the freeze manifest.")
    if asdict(config.prompting) != dict(frozen.get("prompt_settings") or {}):
        raise ValueError("Final configuration prompt does not match the freeze manifest.")

    frozen_model = frozen.get("model")
    if not isinstance(frozen_model, Mapping):
        raise ValueError("Freeze manifest is missing model provenance.")
    shared_model_fields = (
        "backend",
        "model_name_or_path",
        "model_revision",
        "dtype",
        "think",
        "stream",
        "temperature",
        "max_new_tokens",
        "do_sample",
        "trust_remote_code",
    )
    for field in shared_model_fields:
        if field in frozen_model and getattr(config.generation, field) != frozen_model[field]:
            raise ValueError(f"Final generation setting '{field}' does not match the freeze manifest.")
    frozen_batch_size = frozen_model.get("batch_size")
    if frozen_batch_size is not None and config.generation.batch_size != frozen_batch_size:
        equivalence_path = config.final_study.batch_equivalence_manifest.strip()
        if not equivalence_path:
            raise ValueError("Changed execution batch size requires a batch-equivalence manifest.")
        equivalence = _load_json_object(equivalence_path, label="batch-equivalence manifest")
        expected_equivalence = {
            "status": "PASS",
            "freeze_sha256": manifest_sha,
            "model": config.generation.model_name_or_path,
            "source_batch_size": frozen_batch_size,
            "target_batch_size": config.generation.batch_size,
        }
        mismatches = [
            field
            for field, expected in expected_equivalence.items()
            if equivalence.get(field) != expected
        ]
        if mismatches:
            raise ValueError("Batch-equivalence manifest mismatch: " + ", ".join(mismatches))

    configured_methods = set(config.experiments.methods)
    if require_main_methods:
        required_methods = set(PRIMARY_MAIN_METHODS)
        if not required_methods.issubset(configured_methods):
            missing = sorted(required_methods - configured_methods)
            raise ValueError("Final main method set is missing: " + ", ".join(missing))
        prohibited = sorted(configured_methods & {"no_rag", "random_sentence"})
        if prohibited:
            raise ValueError("Primary final main methods include prohibited diagnostics: " + ", ".join(prohibited))
    if require_external_adapter and OPTIONAL_EXTERNAL_METHOD in configured_methods:
        validate_external_compressor_configuration(config.external_compressor)
    return {
        "freeze_sha256": manifest_sha,
        "development_split_sha256": development_sha,
        "final_split_sha256": final_sha,
        "final_count": len(final_ids),
    }


def resolve_final_main_config(
    *,
    template_path: str | Path,
    frozen_config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    require_external_adapter: bool = True,
) -> AppConfig:
    """Deterministically overlay execution-only settings onto the frozen scientific configuration."""
    template = load_config(template_path)
    frozen = load_config(frozen_config_path)
    resolved = replace(
        frozen,
        paths=template.paths,
        logging=template.logging,
        runtime=template.runtime,
        generation=replace(
            frozen.generation,
            device=template.generation.device,
            batch_size=template.generation.batch_size,
        ),
        external_compressor=template.external_compressor,
        final_study=template.final_study,
        experiments=template.experiments,
        robustness=replace(frozen.robustness, supportcover_final_variant="full"),
    )
    validate_final_execution_config(
        resolved,
        manifest_path=manifest_path,
        require_external_adapter=require_external_adapter,
    )
    if output_path is not None:
        write_yaml(output_path, asdict(resolved))
    return resolved


def verify_final_readiness(
    *,
    template_path: str | Path,
    frozen_config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> FinalReadinessReport:
    checks: list[ReadinessCheck] = []
    resolved: AppConfig | None = None
    try:
        manifest = _load_json_object(manifest_path, label="Phase-3 freeze manifest")
        template = load_config(template_path)
        validate_frozen_manifest(manifest, template)
        checks.append(ReadinessCheck("phase3_freeze", True, "Freeze manifest schema and canonical SHA are valid."))
    except Exception as exc:
        checks.append(ReadinessCheck("phase3_freeze", False, str(exc)))
    try:
        resolved = resolve_final_main_config(
            template_path=template_path,
            frozen_config_path=frozen_config_path,
            manifest_path=manifest_path,
            require_external_adapter=False,
        )
        checks.extend(
            (
                ReadinessCheck("final_split_identity", True, "Final count/SHA and development disjointness pass."),
                ReadinessCheck("final_config_resolved", True, "All frozen publication parameters are resolved."),
            )
        )
    except Exception as exc:
        checks.extend(
            (
                ReadinessCheck("final_split_identity", False, str(exc)),
                ReadinessCheck("final_config_resolved", False, str(exc)),
            )
        )
    try:
        method_config = resolved or load_config(template_path)
        configured_methods = set(method_config.experiments.methods)
        missing = set(PRIMARY_MAIN_METHODS) - configured_methods
        prohibited = configured_methods & {"no_rag", "random_sentence"}
        if missing or prohibited:
            raise ValueError(
                "Method-set mismatch; missing="
                + ",".join(sorted(missing))
                + "; prohibited="
                + ",".join(sorted(prohibited))
            )
        checks.append(ReadinessCheck("method_set_complete", True, "All preregistered built-in methods are configured."))
    except Exception as exc:
        checks.append(ReadinessCheck("method_set_complete", False, str(exc)))
    try:
        config_for_external = resolved or load_config(template_path)
        if OPTIONAL_EXTERNAL_METHOD in config_for_external.experiments.methods:
            validate_external_compressor_configuration(config_for_external.external_compressor)
            detail = "External method is configured and its module is available."
        else:
            detail = "External method is not included in this method set."
        checks.append(ReadinessCheck("external_compressor_adapter", True, detail))
    except Exception as exc:
        checks.append(ReadinessCheck("external_compressor_adapter", False, str(exc)))
    statistics_path = Path(__file__).with_name("statistics_artifacts.py")
    checks.append(
        ReadinessCheck(
            "statistics_infrastructure",
            statistics_path.is_file(),
            "Canonical paired-statistics module is present." if statistics_path.is_file() else "Statistics module is missing.",
        )
    )
    try:
        output_dir = Path((resolved or load_config(template_path)).paths.output_root) / "main"
        output_safe = not output_dir.exists() or output_dir.is_dir()
        checks.append(
            ReadinessCheck(
                "output_directory_safe",
                output_safe,
                f"Final run root: {output_dir}" if output_safe else f"Final run root is not a directory: {output_dir}",
            )
        )
    except Exception as exc:
        checks.append(ReadinessCheck("output_directory_safe", False, str(exc)))
    report = FinalReadinessReport(tuple(checks))
    if output_path is not None:
        write_json(output_path, report.to_dict())
    return report


def aggregate_final_main_runs(
    run_dirs: Sequence[str | Path],
    *,
    output_path: str | Path,
    expected_final_count: int,
    expected_final_sha256: str,
    expected_freeze_sha256: str,
    require_external: bool,
) -> list[dict[str, Any]]:
    """Aggregate only complete, provenance-matched final runs; never compute statistics."""
    rows_by_method: dict[str, dict[str, Any]] = {}
    required_methods = list(PRIMARY_MAIN_METHODS)
    if require_external:
        required_methods.append(OPTIONAL_EXTERNAL_METHOD)
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        metrics = _load_json_object(run_dir / "metrics.json", label="final run metrics")
        if metrics.get("status") != "completed":
            raise ValueError(f"Final run is incomplete: {run_dir}")
        method = metrics.get("method")
        if method not in required_methods:
            raise ValueError(f"Unexpected final main method in {run_dir}: {method}")
        if method in rows_by_method:
            raise ValueError(f"Duplicate completed final main method: {method}")
        if metrics.get("num_examples") != expected_final_count:
            raise ValueError(f"Final run count mismatch for {method}.")
        if metrics.get("split_sha256") != expected_final_sha256:
            raise ValueError(f"Final split SHA256 mismatch for {method}.")
        if metrics.get("freeze_sha256") != expected_freeze_sha256:
            raise ValueError(f"Freeze SHA256 mismatch for {method}.")
        config_path = run_dir / "config.resolved.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing resolved config: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            resolved_payload = yaml.safe_load(handle) or {}
        if not isinstance(resolved_payload, dict):
            raise ValueError(f"Resolved config must be a mapping: {config_path}")
        run_metadata = resolved_payload.pop("run", None)
        if not isinstance(run_metadata, Mapping):
            raise ValueError(f"Resolved config is missing run provenance: {config_path}")
        for field in ("experiment_id", "method", "split_sha256", "freeze_sha256", "config_sha256"):
            if run_metadata.get(field) != metrics.get(field):
                raise ValueError(f"Run metrics/config provenance mismatch for {method}: {field}")
        if canonical_sha256(resolved_payload) != metrics.get("config_sha256"):
            raise ValueError(f"Resolved config SHA256 mismatch for {method}.")
        split_payload = resolved_payload.get("split")
        if not isinstance(split_payload, Mapping) or split_payload.get("role") != "final":
            raise ValueError(f"Resolved run is not a final-role configuration: {method}")
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        if len(predictions) != expected_final_count:
            raise ValueError(f"Prediction count mismatch for {method}.")
        prediction_ids = [record.get("example_id") for record in predictions]
        if not all(isinstance(item, str) and item for item in prediction_ids):
            raise ValueError(f"Malformed prediction example IDs for {method}.")
        validate_unique_ids(prediction_ids)
        if any(record.get("method") != method for record in predictions):
            raise ValueError(f"Prediction method mismatch for {method}.")
        support_available = metrics.get("support_f1") is not None
        required_prediction_metrics = ["answer_em", "answer_f1"]
        if support_available:
            required_prediction_metrics.extend(("support_f1", "support_recall", "coverage_at_budget"))
        for row_number, record in enumerate(predictions, start=1):
            for metric in required_prediction_metrics:
                if metric not in record:
                    raise ValueError(f"Prediction row {row_number} for {method} is missing {metric}.")
                value = _require_number(record[metric], label=f"{method} prediction {metric}")
                if metric == "answer_em" and value not in (0.0, 1.0):
                    raise ValueError(f"Prediction row {row_number} for {method} has non-binary answer_em.")
        config_id = str(metrics.get("config_id") or method)
        if method == OPTIONAL_EXTERNAL_METHOD:
            external = resolved_payload.get("external_compressor") or {}
            implementation_id = external.get("implementation_id")
            if not implementation_id:
                raise ValueError("External compressor final run lacks implementation identity.")
            expected_config_id = f"external_compressor_{implementation_id}"
            if config_id != expected_config_id:
                raise ValueError("External compressor config_id does not match implementation identity.")
        row = {
            "method": method,
            "config_id": config_id,
            "num_examples": expected_final_count,
            **{field: metrics.get(field) for field in _RESULT_FIELDS},
            "support_metrics_available": support_available,
            "run_directory": str(run_dir),
            "split_sha256": expected_final_sha256,
            "freeze_sha256": expected_freeze_sha256,
            "config_sha256": metrics.get("config_sha256"),
            "experiment_id": metrics.get("experiment_id"),
            "code_revision": metrics.get("code_revision"),
        }
        rows_by_method[method] = row
    missing = [method for method in required_methods if method not in rows_by_method]
    if missing:
        raise ValueError("Missing complete final main runs: " + ", ".join(missing))
    ordered_rows = [rows_by_method[method] for method in required_methods]
    write_csv(output_path, ordered_rows)
    return ordered_rows
