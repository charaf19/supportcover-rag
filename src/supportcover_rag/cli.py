from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import typer

from supportcover_rag.config import AppConfig, load_config
from supportcover_rag.data import acquire_hotpotqa, preprocess_raw_split
from supportcover_rag.error_analysis import run_error_analysis as generate_error_analysis
from supportcover_rag.experiment_outputs import ExperimentFamily, VALID_EXPERIMENT_FAMILIES, parse_experiment_family
from supportcover_rag.logging_utils import configure_logging
from supportcover_rag.pipeline import ExperimentRunner, SUPPORTED_METHODS
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
    family: str | None = typer.Option(
        None,
        help="Experiment family: " + "|".join(VALID_EXPERIMENT_FAMILIES) + ". Defaults to main.",
    ),
    notes: str = typer.Option("", help="Optional notes recorded in the experiment registry."),
    experiment_id: str | None = typer.Option(None, help="Optional explicit experiment id, for example: EXP001 or DBG001."),
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
    )
    resolved_split = _resolve_split(split, cfg)
    split_path = Path(cfg.paths.data_root) / "processed" / f"{resolved_split}.jsonl"
    runner = ExperimentRunner(cfg)
    runner.run_main_suite(
        split_path=split_path,
        split_name=resolved_split,
        family=_parse_family(family, default=ExperimentFamily.MAIN) or ExperimentFamily.MAIN,
        notes=notes,
        experiment_id=experiment_id,
    )
    LOGGER.info("Main experiment suite complete.")


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
