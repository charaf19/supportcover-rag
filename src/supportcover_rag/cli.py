from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import typer

from supportcover_rag.config import AppConfig, load_config
from supportcover_rag.data import acquire_hotpotqa, preprocess_raw_split
from supportcover_rag.error_analysis import run_error_analysis as generate_error_analysis
from supportcover_rag.environment import collect_environment_manifest
from supportcover_rag.experiment_outputs import ExperimentFamily, VALID_EXPERIMENT_FAMILIES, parse_experiment_family
from supportcover_rag.external_baselines import load_configured_external_compressor
from supportcover_rag.final_execution import (
    OPTIONAL_EXTERNAL_METHOD,
    aggregate_final_main_runs,
    resolve_final_main_config,
    validate_final_execution_config,
    verify_final_readiness,
)
from supportcover_rag.io_utils import read_jsonl, write_json
from supportcover_rag.logging_utils import configure_logging
from supportcover_rag.pipeline import ExperimentRunner, SUPPORTED_METHODS
from supportcover_rag.paper_artifacts import export_development_paper_results, export_protocol_paper_results
from supportcover_rag.phase3 import (
    aggregate_development_generation,
    freeze_development_selection,
    run_packing_screen,
    validate_development_protocol,
)
from supportcover_rag.robustness import aggregate_robustness_runs, verify_robustness_readiness
from supportcover_rag.splits import (
    build_record_strata,
    build_split_manifest,
    load_json_ids,
    ordered_ids_sha256,
    select_seeded_stratified_ids,
    validate_disjoint_splits,
    validate_unique_ids,
)
from supportcover_rag.statistics_artifacts import run_statistics_plan
from supportcover_rag.systems_summary import run_systems_summary as generate_systems_summary

app = typer.Typer(add_completion=False, no_args_is_help=True)
LOGGER = logging.getLogger(__name__)
SUPPORTED_BACKENDS = ("transformers", "ollama")


def _parse_methods(methods: str | None) -> list[str] | None:
    if methods is None:
        return None

    selected_methods: list[str] = []
    seen_methods: set[str] = set()
    for raw_method in methods.split(","):
        method = raw_method.strip()
        if not method:
            continue
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(SUPPORTED_METHODS)
            raise typer.BadParameter(f"Unsupported method '{method}'. Expected one of: {supported}.")
        if method in seen_methods:
            continue
        selected_methods.append(method)
        seen_methods.add(method)

    if not selected_methods:
        raise typer.BadParameter("Provide at least one method in --methods, for example: supportcover,relevance_only")
    return selected_methods


def _parse_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise typer.BadParameter(f"Unsupported backend '{backend}'. Expected one of: {supported}.")
    return normalized


def _parse_family(
    family: str | None,
    *,
    default: ExperimentFamily | None = None,
) -> ExperimentFamily | None:
    try:
        return parse_experiment_family(family, default=default)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_split(split: str | None, config: AppConfig) -> str:
    candidate = split.strip() if split is not None else config.experiments.split.strip()
    if not candidate:
        raise typer.BadParameter("Experiment split cannot be empty.")
    if config.split.role.strip().lower() == "development":
        if candidate.lower() != "train":
            raise typer.BadParameter("role=development is locked to the processed train split.")
        try:
            validate_development_protocol(config)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif config.split.role.strip().lower() == "final":
        try:
            validate_final_execution_config(
                config,
                require_external_adapter=False,
                require_main_methods=False,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    return candidate


def _apply_overrides(
    config: AppConfig,
    *,
    backend: str | None,
    model: str | None,
    ollama_base_url: str | None,
    device: str | None,
    dtype: str | None,
    limit: int | None,
    methods: list[str] | None,
    max_new_tokens: int | None,
    batch_size: int | None,
    beta_coverage: float | None,
    title_bonus: float | None,
    delta_token_cost: float | None,
    gamma_redundancy: float | None,
    mmr_lambda: float | None,
) -> AppConfig:
    if any(value is not None for value in (backend, model, ollama_base_url, device, dtype, max_new_tokens, batch_size)):
        config = replace(
            config,
            generation=replace(
                config.generation,
                backend=backend if backend is not None else config.generation.backend,
                model_name_or_path=model if model is not None else config.generation.model_name_or_path,
                base_url=ollama_base_url if ollama_base_url is not None else config.generation.base_url,
                device=device if device is not None else config.generation.device,
                dtype=dtype if dtype is not None else config.generation.dtype,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else config.generation.max_new_tokens,
                batch_size=batch_size if batch_size is not None else config.generation.batch_size,
            ),
        )
    if limit is not None:
        config = replace(config, runtime=replace(config.runtime, limit=limit))
    if methods is not None:
        config = replace(config, experiments=replace(config.experiments, methods=methods))
    tuning_override_requested = any(
        value is not None
        for value in (beta_coverage, title_bonus, delta_token_cost, gamma_redundancy, mmr_lambda)
    )
    if tuning_override_requested and config.split.role.strip().lower() != "development":
        raise typer.BadParameter("Coefficient and MMR overrides are permitted only for role=development.")
    if any(value is not None for value in (beta_coverage, title_bonus, delta_token_cost, gamma_redundancy)):
        config = replace(
            config,
            supportcover=replace(
                config.supportcover,
                beta_coverage=beta_coverage if beta_coverage is not None else config.supportcover.beta_coverage,
                title_bonus=title_bonus if title_bonus is not None else config.supportcover.title_bonus,
                delta_token_cost=(
                    delta_token_cost if delta_token_cost is not None else config.supportcover.delta_token_cost
                ),
                gamma_redundancy=(
                    gamma_redundancy if gamma_redundancy is not None else config.supportcover.gamma_redundancy
                ),
            ),
        )
    if mmr_lambda is not None:
        if not 0.0 <= mmr_lambda <= 1.0:
            raise typer.BadParameter("--mmr-lambda must be between 0 and 1.")
        config = replace(config, retrieval=replace(config.retrieval, mmr_lambda_relevance=mmr_lambda))
    return config


def _load_app_config(
    config_path: str,
    *,
    backend: str | None = None,
    model: str | None = None,
    ollama_base_url: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    limit: int | None = None,
    methods: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
    beta_coverage: float | None = None,
    title_bonus: float | None = None,
    delta_token_cost: float | None = None,
    gamma_redundancy: float | None = None,
    mmr_lambda: float | None = None,
) -> tuple[Path, AppConfig]:
    config = load_config(config_path)
    config = _apply_overrides(
        config,
        backend=_parse_backend(backend),
        model=model,
        ollama_base_url=ollama_base_url,
        device=device,
        dtype=dtype,
        limit=limit,
        methods=_parse_methods(methods),
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        beta_coverage=beta_coverage,
        title_bonus=title_bonus,
        delta_token_cost=delta_token_cost,
        gamma_redundancy=gamma_redundancy,
        mmr_lambda=mmr_lambda,
    )
    configure_logging(config.logging.level)
    return Path(config_path), config


@app.command("acquire-data")
def acquire_data(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    _, cfg = _load_app_config(config)
    raw_dir = Path(cfg.paths.data_root) / "raw"
    acquire_hotpotqa(
        dataset_path=cfg.raw_data.dataset_path,
        dataset_config=cfg.raw_data.dataset_config,
        splits=cfg.raw_data.splits,
        output_dir=raw_dir,
    )
    LOGGER.info("Raw data acquisition complete: %s", raw_dir)


@app.command("preprocess")
def preprocess(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    _, cfg = _load_app_config(config)
    raw_dir = Path(cfg.paths.data_root) / "raw"
    processed_dir = Path(cfg.paths.data_root) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in cfg.raw_data.splits:
        preprocess_raw_split(
            raw_path=raw_dir / f"{split}.jsonl",
            processed_path=processed_dir / f"{split}.jsonl",
            limit=cfg.runtime.limit,
        )
    LOGGER.info("Preprocessing complete: %s", processed_dir)


@app.command("create-split")
def create_split(
    processed: str = typer.Option(..., help="Path to a processed JSONL population."),
    output: str = typer.Option(..., help="Path for the self-contained JSON split manifest."),
    role: str = typer.Option(..., help="Scientific role, for example development or final."),
    sample_size: int | None = typer.Option(None, min=0, help="Number of IDs; omit to select the full population."),
    seed: int = typer.Option(42, help="Deterministic sampling seed."),
    stratify_by: str = typer.Option("", help="Comma-separated processed-record fields, for example type,level."),
) -> None:
    rows = read_jsonl(processed)
    ids: list[str] = []
    for index, row in enumerate(rows):
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise typer.BadParameter(f"Processed row {index} is missing a non-empty string ID.")
        ids.append(item_id)
    validate_unique_ids(ids)

    dimensions = [dimension.strip() for dimension in stratify_by.split(",") if dimension.strip()]
    strata = build_record_strata(rows, dimensions) if dimensions else None
    resolved_sample_size = len(ids) if sample_size is None else sample_size
    selected_ids = select_seeded_stratified_ids(
        ids,
        sample_size=resolved_sample_size,
        seed=seed,
        strata=strata,
    )
    manifest = build_split_manifest(
        selected_ids,
        ids_file=output,
        role=role,
        seed=seed,
        stratify_by=dimensions,
        source_path=processed,
    )
    write_json(output, manifest)
    typer.echo(
        f"Wrote {manifest['count']} {manifest['role']} IDs to {output} "
        f"(SHA256 {manifest['split_sha256']})."
    )


@app.command("validate-splits")
def validate_splits(
    development: str = typer.Option(..., help="Path to the development split JSON."),
    final: str = typer.Option(..., help="Path to the final split JSON."),
    output: str | None = typer.Option(None, help="Optional path for the validation report JSON."),
) -> None:
    development_ids = load_json_ids(development)
    final_ids = load_json_ids(final)
    validate_disjoint_splits({"development": development_ids, "final": final_ids})
    report = {
        "status": "PASS",
        "development": {
            "ids_file": development,
            "count": len(development_ids),
            "split_sha256": ordered_ids_sha256(development_ids),
        },
        "final": {
            "ids_file": final,
            "count": len(final_ids),
            "split_sha256": ordered_ids_sha256(final_ids),
        },
        "overlap_count": 0,
    }
    if output is not None:
        write_json(output, report)
    typer.echo(
        "PASS: development and final splits are disjoint "
        f"({len(development_ids)} development, {len(final_ids)} final IDs)."
    )


@app.command("run-development-packing")
def run_development_packing(
    config: str = typer.Option(..., help="Path to the development-only Phase-3 YAML config."),
) -> None:
    _, cfg = _load_app_config(config)
    manifest = run_packing_screen(cfg)
    typer.echo(
        "Completed generator-free Phase-3 packing screen on "
        f"{manifest['development_count']} development examples "
        f"({manifest['num_plan_items']} configurations)."
    )


@app.command("check-environment")
def check_environment(
    output: str = typer.Option(
        "outputs/development/phase3/environment.json",
        help="Path for the machine-readable runtime manifest.",
    ),
) -> None:
    manifest = collect_environment_manifest(metadata={"phase": 3, "scientific_role": "development"})
    write_json(output, manifest)
    accelerator = manifest["accelerator"]
    typer.echo(
        f"Environment recorded at {output}: "
        f"torch={manifest['packages']['torch']}, "
        f"accelerator={accelerator['backend'] or 'none'}, "
        f"device={accelerator['device_name'] or 'none'}."
    )


@app.command("freeze-development")
def freeze_development(
    config: str = typer.Option(..., help="Path to the development-only Phase-3 YAML config."),
    decision: str = typer.Option(..., help="Development evidence and selected-parameter decision JSON."),
    final_ids: str = typer.Option(
        "data/splits/final_ids.json",
        help="Frozen final ID manifest; only IDs/count/SHA are read, never final examples or predictions.",
    ),
    final_config: str = typer.Option("configs/final_frozen.yaml", help="Output frozen final YAML config."),
    manifest: str = typer.Option(
        "configs/frozen/final_manifest.json",
        help="Output deterministic freeze manifest.",
    ),
) -> None:
    _, cfg = _load_app_config(config)
    frozen = freeze_development_selection(
        cfg,
        decision_path=decision,
        final_ids_path=final_ids,
        final_config_path=final_config,
        manifest_path=manifest,
    )
    typer.echo(f"Frozen Phase-3 configuration SHA256: {frozen['config_sha256']}")


@app.command("aggregate-development-generation")
def aggregate_development_generation_command(
    shortlist: str = typer.Option(
        "outputs/development/phase3/shortlist.json",
        help="Recorded Phase-3 shortlist and generation plan.",
    ),
    output: str = typer.Option(
        "outputs/development/phase3/generation_validation.csv",
        help="Compact development-generation evidence CSV.",
    ),
) -> None:
    rows = aggregate_development_generation(shortlist_path=shortlist, output_path=output)
    typer.echo(f"Wrote {len(rows)} validated development-generation rows to {output}.")


@app.command("export-paper-development")
def export_paper_development(
    packing_summary: str = typer.Option(
        "outputs/development/phase3/packing/packing_summary.csv",
        help="Raw Phase-3 packing summary CSV.",
    ),
    packing_manifest: str = typer.Option(
        "outputs/development/phase3/packing/packing_manifest.json",
        help="Raw Phase-3 packing manifest.",
    ),
    shortlist: str = typer.Option(
        "outputs/development/phase3/shortlist.json",
        help="Recorded development shortlist.",
    ),
    decision: str = typer.Option(
        "outputs/development/phase3/decision.json",
        help="Completed development decision record.",
    ),
    freeze_manifest: str = typer.Option(
        "configs/frozen/final_manifest.json",
        help="Completed deterministic freeze manifest.",
    ),
    output_root: str = typer.Option("paper_results", help="Publication-only artifact root."),
    code_revision: str | None = typer.Option(None, help="Optional source-control revision recorded as provenance."),
) -> None:
    artifacts = export_development_paper_results(
        packing_summary_path=packing_summary,
        packing_manifest_path=packing_manifest,
        shortlist_path=shortlist,
        decision_path=decision,
        freeze_manifest_path=freeze_manifest,
        output_root=output_root,
        code_revision=code_revision,
    )
    typer.echo("Exported curated development artifacts: " + ", ".join(artifacts.values()))


@app.command("export-paper-protocol")
def export_paper_protocol(
    split_validation: str = typer.Option(
        "data/splits/split_validation.json",
        help="Verified development/final split-validation report.",
    ),
    frozen_config: str = typer.Option("configs/final_frozen.yaml", help="Completed frozen configuration."),
    freeze_manifest: str = typer.Option(
        "configs/frozen/final_manifest.json",
        help="Completed deterministic freeze manifest.",
    ),
    environment: str = typer.Option(
        "outputs/development/phase3/environment.json",
        help="Verified Phase-3 environment manifest.",
    ),
    output_root: str = typer.Option("paper_results", help="Publication-only artifact root."),
    code_revision: str | None = typer.Option(None, help="Optional source-control revision recorded as provenance."),
) -> None:
    artifacts = export_protocol_paper_results(
        split_validation_path=split_validation,
        frozen_config_path=frozen_config,
        freeze_manifest_path=freeze_manifest,
        environment_path=environment,
        output_root=output_root,
        code_revision=code_revision,
    )
    typer.echo("Exported frozen protocol artifacts: " + ", ".join(artifacts.values()))


@app.command("run")
def run(
    config: str = typer.Option(..., help="Path to YAML config."),
    split: str | None = typer.Option(None, help="Data split to run. Defaults to experiments.split from the config."),
    backend: str | None = typer.Option(None, help="Override generation backend: transformers or ollama."),
    model: str | None = typer.Option(None, help="Override generation model name or path."),
    ollama_base_url: str | None = typer.Option(None, help="Override Ollama base URL, for example: http://localhost:11434."),
    device: str | None = typer.Option(None, help="Override generation device: auto, xpu, cpu, cuda, or mps."),
    dtype: str | None = typer.Option(None, help="Override generation dtype: auto, float32, or float16."),
    methods: str | None = typer.Option(None, help="Comma-separated method list override, for example: supportcover,relevance_only."),
    limit: int | None = typer.Option(None, min=1, help="Limit the number of processed examples at runtime."),
    max_new_tokens: int | None = typer.Option(None, min=1, help="Override generation.max_new_tokens."),
    batch_size: int | None = typer.Option(None, min=1, help="Override generation.batch_size."),
    beta_coverage: float | None = typer.Option(None, help="Development-only SupportCover beta override."),
    title_bonus: float | None = typer.Option(None, help="Development-only SupportCover title-bonus override."),
    delta_token_cost: float | None = typer.Option(None, help="Development-only SupportCover token-cost override."),
    gamma_redundancy: float | None = typer.Option(None, help="Development-only SupportCover redundancy override."),
    mmr_lambda: float | None = typer.Option(None, help="Development-only MMR lambda override."),
    family: str | None = typer.Option(
        None,
        help="Experiment family: " + "|".join(VALID_EXPERIMENT_FAMILIES) + ". Defaults to main.",
    ),
    notes: str = typer.Option("", help="Optional notes recorded in the experiment registry."),
    experiment_id: str | None = typer.Option(None, help="Optional explicit experiment id, for example: EXP001 or DBG001."),
    code_revision: str | None = typer.Option(None, help="Optional source-control revision recorded as provenance."),
) -> None:
    _, cfg = _load_app_config(
        config,
        backend=backend,
        model=model,
        ollama_base_url=ollama_base_url,
        device=device,
        dtype=dtype,
        limit=limit,
        methods=methods,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        beta_coverage=beta_coverage,
        title_bonus=title_bonus,
        delta_token_cost=delta_token_cost,
        gamma_redundancy=gamma_redundancy,
        mmr_lambda=mmr_lambda,
    )
    resolved_split = _resolve_split(split, cfg)
    if cfg.split.role.strip().lower() == "final":
        validate_final_execution_config(cfg)
    split_path = Path(cfg.paths.data_root) / "processed" / f"{resolved_split}.jsonl"
    external_compressor = (
        load_configured_external_compressor(cfg.external_compressor)
        if OPTIONAL_EXTERNAL_METHOD in cfg.experiments.methods
        else None
    )
    runner = ExperimentRunner(cfg, external_compressor=external_compressor)
    runner.run_main_suite(
        split_path=split_path,
        split_name=resolved_split,
        family=_parse_family(family, default=ExperimentFamily.MAIN) or ExperimentFamily.MAIN,
        notes=notes,
        experiment_id=experiment_id,
        code_revision=code_revision,
    )
    LOGGER.info("Main experiment suite complete.")


@app.command("verify-final-readiness")
def verify_final_readiness_command(
    template: str = typer.Option("configs/final_main.yaml", help="Unresolved final-main template."),
    frozen_config: str = typer.Option("configs/final_frozen.yaml", help="Phase-3 frozen configuration."),
    manifest: str = typer.Option("configs/frozen/final_manifest.json", help="Phase-3 freeze manifest."),
    output: str | None = typer.Option(None, help="Optional metadata-only readiness report JSON."),
) -> None:
    report = verify_final_readiness(
        template_path=template,
        frozen_config_path=frozen_config,
        manifest_path=manifest,
        output_path=output,
    )
    for check in report.checks:
        typer.echo(f"{check.name}: {'PASS' if check.passed else 'FAIL'} - {check.detail}")
    typer.echo(f"FINAL MAIN STUDY: {'READY' if report.ready else 'BLOCKED'}")


@app.command("resolve-final-main")
def resolve_final_main_command(
    template: str = typer.Option("configs/final_main.yaml", help="Unresolved final-main template."),
    frozen_config: str = typer.Option("configs/final_frozen.yaml", help="Phase-3 frozen configuration."),
    manifest: str = typer.Option("configs/frozen/final_manifest.json", help="Phase-3 freeze manifest."),
    output: str = typer.Option("configs/final_main.resolved.yaml", help="Resolved final-main configuration."),
) -> None:
    resolve_final_main_config(
        template_path=template,
        frozen_config_path=frozen_config,
        manifest_path=manifest,
        output_path=output,
    )
    typer.echo(f"Resolved final main configuration: {output}")


@app.command("aggregate-final-main")
def aggregate_final_main_command(
    run_dir: list[str] = typer.Option(..., "--run-dir", help="Completed final run directory; repeat per method."),
    config: str = typer.Option("configs/final_main.resolved.yaml", help="Resolved final-main configuration."),
    output: str = typer.Option("outputs/final/main_results.csv", help="Raw final main summary CSV."),
) -> None:
    cfg = load_config(config)
    identity = validate_final_execution_config(cfg)
    rows = aggregate_final_main_runs(
        run_dir,
        output_path=output,
        expected_final_count=int(identity["final_count"]),
        expected_final_sha256=str(identity["final_split_sha256"]),
        expected_freeze_sha256=str(identity["freeze_sha256"]),
        require_external=OPTIONAL_EXTERNAL_METHOD in cfg.experiments.methods,
    )
    typer.echo(f"Aggregated {len(rows)} complete final main runs: {output}")


@app.command("run-statistics")
def run_statistics(
    plan: str = typer.Option(..., help="Path to a paired-statistics plan JSON."),
    output_dir: str | None = typer.Option(
        None,
        help="Optional output directory override. Defaults to outputs/statistics under the plan project root.",
    ),
) -> None:
    artifacts = run_statistics_plan(plan, output_dir=output_dir)
    typer.echo(
        "Completed paired statistics artifacts: "
        f"{artifacts['method_summary']}, {artifacts['paired_comparisons']}, "
        f"{artifacts['statistics_manifest']}"
    )


@app.command("run-ablations")
def run_ablations(
    config: str = typer.Option(..., help="Path to YAML config."),
    split: str | None = typer.Option(None, help="Data split to run. Defaults to experiments.split from the config."),
    backend: str | None = typer.Option(None, help="Override generation backend: transformers or ollama."),
    model: str | None = typer.Option(None, help="Override generation model name or path."),
    ollama_base_url: str | None = typer.Option(None, help="Override Ollama base URL, for example: http://localhost:11434."),
    device: str | None = typer.Option(None, help="Override generation device: auto, xpu, cpu, cuda, or mps."),
    dtype: str | None = typer.Option(None, help="Override generation dtype: auto, float32, or float16."),
    limit: int | None = typer.Option(None, min=1, help="Limit the number of processed examples at runtime."),
    max_new_tokens: int | None = typer.Option(None, min=1, help="Override generation.max_new_tokens."),
    batch_size: int | None = typer.Option(None, min=1, help="Override generation.batch_size."),
    beta_coverage: float | None = typer.Option(None, help="Development-only SupportCover beta override."),
    title_bonus: float | None = typer.Option(None, help="Development-only SupportCover title-bonus override."),
    delta_token_cost: float | None = typer.Option(None, help="Development-only SupportCover token-cost override."),
    gamma_redundancy: float | None = typer.Option(None, help="Development-only SupportCover redundancy override."),
    family: str | None = typer.Option(
        None,
        help=(
            "Optional family override. Use ablation_budget, ablation_depth, or ablation_component to run one sweep, "
            "or debug to route all ablations under debug/. Defaults to all ablation families."
        ),
    ),
    notes: str = typer.Option("", help="Optional notes recorded in the experiment registry."),
    experiment_id: str | None = typer.Option(None, help="Optional explicit experiment id when only one run will be produced."),
) -> None:
    _, cfg = _load_app_config(
        config,
        backend=backend,
        model=model,
        ollama_base_url=ollama_base_url,
        device=device,
        dtype=dtype,
        limit=limit,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        beta_coverage=beta_coverage,
        title_bonus=title_bonus,
        delta_token_cost=delta_token_cost,
        gamma_redundancy=gamma_redundancy,
    )
    resolved_split = _resolve_split(split, cfg)
    split_path = Path(cfg.paths.data_root) / "processed" / f"{resolved_split}.jsonl"
    runner = ExperimentRunner(cfg)
    runner.run_ablations(
        split_path=split_path,
        split_name=resolved_split,
        family=_parse_family(family),
        notes=notes,
        experiment_id=experiment_id,
    )
    LOGGER.info("Ablation suite complete.")


@app.command("run-robustness")
def run_robustness(
    config: str = typer.Option(..., help="Path to YAML config."),
    split: str | None = typer.Option(None, help="Data split to run. Defaults to experiments.split from the config."),
    device: str | None = typer.Option(None, help="Override generation device: auto, xpu, cpu, cuda, or mps."),
    dtype: str | None = typer.Option(None, help="Override generation dtype: auto, float32, or float16."),
    limit: int | None = typer.Option(None, min=1, help="Limit the number of processed examples at runtime."),
    batch_size: int | None = typer.Option(None, min=1, help="Override generation.batch_size."),
    notes: str = typer.Option("", help="Optional notes recorded in the experiment registry."),
) -> None:
    raise typer.BadParameter(
        "The historical ungated run-robustness command is disabled. "
        "Phase-6 execution must use a validated family plan after the Phase-3 freeze "
        "and Phase-5 main-study completion."
    )


@app.command("verify-robustness-readiness")
def verify_robustness_readiness_command(
    freeze_manifest: str = typer.Option(
        "configs/frozen/final_manifest.json", help="Phase-3 freeze manifest."
    ),
    main_completion: str = typer.Option(
        "outputs/final/main_study_completion.json", help="Phase-5 completion manifest."
    ),
    budget_plan: str = typer.Option(
        "configs/robustness_budget.template.yaml", help="Budget-robustness plan."
    ),
    model_plan: str = typer.Option(
        "configs/robustness_models.template.yaml", help="Model-robustness plan."
    ),
    cross_dataset_plan: str = typer.Option(
        "configs/robustness_cross_dataset.template.yaml", help="Cross-dataset plan."
    ),
    output: str | None = typer.Option(None, help="Optional metadata-only readiness report JSON."),
) -> None:
    report = verify_robustness_readiness(
        freeze_manifest_path=freeze_manifest,
        main_completion_path=main_completion,
        budget_plan_path=budget_plan,
        model_plan_path=model_plan,
        cross_dataset_plan_path=cross_dataset_plan,
        output_path=output,
    )
    for check in report.checks:
        typer.echo(f"{check.name}: {'PASS' if check.passed else 'FAIL'} - {check.detail}")
    typer.echo(f"ROBUSTNESS EXECUTION: {'READY' if report.ready else 'BLOCKED'}")


@app.command("aggregate-robustness")
def aggregate_robustness_command(
    family: str = typer.Option(..., help="Robustness family: budget, models, or cross_dataset."),
    run_dir: list[str] = typer.Option(..., "--run-dir", help="Completed run directory; repeat per run."),
    output: str = typer.Option(..., help="Family-specific aggregate CSV under outputs/final/robustness."),
) -> None:
    rows = aggregate_robustness_runs(family, run_dir, output_path=output)
    typer.echo(f"Aggregated {len(rows)} completed {family} robustness runs: {output}")


@app.command("run-robustness-legacy-disabled")
def run_robustness_legacy_disabled(
    config: str = typer.Option(..., help="Historical config; retained only for traceability."),
) -> None:
    """Never executes: legacy robustness did not enforce the scientific freeze boundary."""
    raise typer.BadParameter(
        "Legacy robustness execution is intentionally disabled; use Phase-6 family plans after readiness passes."
    )


def _historical_run_robustness_unreachable(
    config: str,
    split: str | None,
    device: str | None,
    dtype: str | None,
    limit: int | None,
    batch_size: int | None,
    notes: str,
) -> None:
    """Preserve the old implementation for source archaeology, not execution."""
    _, cfg = _load_app_config(
        config,
        device=device,
        dtype=dtype,
        limit=limit,
        batch_size=batch_size,
    )
    resolved_split = _resolve_split(split, cfg)
    split_path = Path(cfg.paths.data_root) / "processed" / f"{resolved_split}.jsonl"
    ExperimentRunner.run_robustness_study(
        cfg,
        split_path=split_path,
        split_name=resolved_split,
        notes=notes,
    )
    LOGGER.info("Robustness suite complete.")


@app.command("run-error-analysis")
def run_error_analysis(
    config: str = typer.Option(..., help="Path to YAML config."),
) -> None:
    _, cfg = _load_app_config(config)
    artifacts = generate_error_analysis(cfg)
    LOGGER.info(
        "Error analysis complete | annotations=%s | summary=%s | analysis=%s",
        artifacts["annotation_path"],
        artifacts["summary_path"],
        artifacts["analysis_path"],
    )


@app.command("run-systems-summary")
def run_systems_summary(
    config: str = typer.Option(..., help="Path to YAML config."),
) -> None:
    _, cfg = _load_app_config(config)
    artifacts = generate_systems_summary(cfg)
    LOGGER.info(
        "Systems summary complete | summary_csv=%s | summary_md=%s | analysis=%s | latency_breakdown=%s",
        artifacts["summary_csv_path"],
        artifacts["summary_md_path"],
        artifacts["analysis_path"],
        artifacts["latency_breakdown_path"],
    )
