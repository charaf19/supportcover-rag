from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from supportcover_rag.config import AppConfig
from supportcover_rag.io_utils import ensure_dir, write_csv


@dataclass(slots=True)
class SystemsSummarySource:
    method: str
    run_dir: Path
    metrics_path: Path
    experiment_id: str
    metrics: dict[str, object]


def _load_source(method: str, run_dir: str | Path) -> SystemsSummarySource:
    source_dir = Path(run_dir)
    metrics_path = source_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics artifact: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(metrics.get("status", "")) != "completed":
        raise ValueError(f"Systems summary expects completed runs only: {metrics_path}")
    if str(metrics.get("method", "")) != method:
        raise ValueError(
            f"Configured method '{method}' does not match metrics method '{metrics.get('method')}' in {metrics_path}."
        )
    return SystemsSummarySource(
        method=method,
        run_dir=source_dir,
        metrics_path=metrics_path,
        experiment_id=str(metrics.get("experiment_id", source_dir.name.split("_", 1)[0])),
        metrics=metrics,
    )


def _shared_value(sources: list[SystemsSummarySource], key: str) -> object:
    values = {source.metrics.get(key) for source in sources}
    if len(values) != 1:
        raise ValueError(f"Phase 8 expects a frozen setup; found multiple values for '{key}': {sorted(values)}")
    return next(iter(values))


def _build_summary_rows(config: AppConfig) -> list[dict[str, object]]:
    settings = config.systems_summary
    methods = [settings.comparator_method, settings.canonical_method]
    missing_methods = [method for method in methods if method not in settings.source_runs]
    if missing_methods:
        missing = ", ".join(missing_methods)
        raise ValueError(f"Missing source_runs entries for: {missing}")

    sources = [_load_source(method, settings.source_runs[method]) for method in methods]
    _shared_value(sources, "model")
    _shared_value(sources, "token_budget")
    _shared_value(sources, "retrieval_depth")
    _shared_value(sources, "num_examples")

    baseline_metrics = sources[0].metrics
    baseline_non_generation = float(baseline_metrics["retrieval_latency_ms"]) + float(baseline_metrics["packing_latency_ms"])
    baseline_peak_rss = float(baseline_metrics["peak_rss_mb"])

    rows: list[dict[str, object]] = []
    for source in sources:
        metrics = source.metrics
        retrieval_latency_ms = float(metrics["retrieval_latency_ms"])
        packing_latency_ms = float(metrics["packing_latency_ms"])
        generation_latency_ms = float(metrics["generation_latency_ms"])
        total_latency_ms = float(metrics["total_latency_ms"])
        total_non_generation_latency_ms = retrieval_latency_ms + packing_latency_ms
        retrieval_pct = 0.0 if total_latency_ms == 0 else (retrieval_latency_ms / total_latency_ms) * 100.0
        packing_pct = 0.0 if total_latency_ms == 0 else (packing_latency_ms / total_latency_ms) * 100.0
        generation_pct = 0.0 if total_latency_ms == 0 else (generation_latency_ms / total_latency_ms) * 100.0
        overhead_ms = total_non_generation_latency_ms - baseline_non_generation
        overhead_pct = 0.0 if baseline_non_generation == 0 else (overhead_ms / baseline_non_generation) * 100.0
        memory_delta_mb = float(metrics["peak_rss_mb"]) - baseline_peak_rss
        memory_delta_pct = 0.0 if baseline_peak_rss == 0 else (memory_delta_mb / baseline_peak_rss) * 100.0

        rows.append(
            {
                "method": str(metrics["method"]),
                "variant": str(metrics["variant"]),
                "model": str(metrics["model"]),
                "answer_f1": float(metrics["answer_f1"]),
                "support_f1": float(metrics["support_f1"]),
                "coverage_at_budget": float(metrics["coverage_at_budget"]),
                "evidence_tokens": float(metrics["evidence_tokens"]),
                "retrieval_latency_ms": retrieval_latency_ms,
                "packing_latency_ms": packing_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "total_non_generation_latency_ms": total_non_generation_latency_ms,
                "total_latency_ms": total_latency_ms,
                "retrieval_pct_of_total": retrieval_pct,
                "packing_pct_of_total": packing_pct,
                "generation_pct_of_total": generation_pct,
                "supportcover_overhead_vs_relevance_only_ms": overhead_ms,
                "supportcover_overhead_vs_relevance_only_pct": overhead_pct,
                "peak_rss_mb": float(metrics["peak_rss_mb"]),
                "memory_delta_mb": memory_delta_mb,
                "memory_delta_pct": memory_delta_pct,
                "num_examples": int(metrics["num_examples"]),
                "token_budget": int(metrics["token_budget"]),
                "retrieval_depth": int(metrics["retrieval_depth"]),
                "experiment_id": str(metrics["experiment_id"]),
                "output_dir": str(metrics["output_dir"]),
            }
        )
    return rows


def _build_latency_breakdown_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in summary_rows:
        for component in ("retrieval", "packing", "generation"):
            rows.append(
                {
                    "method": summary["method"],
                    "variant": summary["variant"],
                    "model": summary["model"],
                    "latency_component": component,
                    "latency_ms": summary[f"{component}_latency_ms"],
                    "pct_of_total": summary[f"{component}_pct_of_total"],
                    "experiment_id": summary["experiment_id"],
                    "output_dir": summary["output_dir"],
                }
            )
    return rows


def _format_float(value: object, decimals: int = 2) -> str:
    return f"{float(value):.{decimals}f}"


def _render_summary_markdown(
    *,
    config: AppConfig,
    summary_rows: list[dict[str, object]],
) -> str:
    settings = config.systems_summary
    lines = [
        "# Phase 8 Systems Summary",
        "",
        "Phase 8 reuses the frozen Qwen main comparison from Phase 6 and summarizes only `relevance_only` versus `supportcover_final`, where `supportcover_final = no_redundancy`.",
        "",
        f"Frozen setup config: `{settings.frozen_setup_config}`",
        f"- comparator source: `{settings.source_runs[settings.comparator_method]}`",
        f"- canonical source: `{settings.source_runs[settings.canonical_method]}`",
        "",
        "| method | variant | model | answer_f1 | support_f1 | coverage_at_budget | evidence_tokens | retrieval_latency_ms | packing_latency_ms | generation_latency_ms | total_non_generation_latency_ms | total_latency_ms | retrieval_pct_of_total | packing_pct_of_total | generation_pct_of_total | supportcover_overhead_vs_relevance_only_ms | peak_rss_mb | experiment_id | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["variant"]),
                    str(row["model"]),
                    _format_float(row["answer_f1"], 4),
                    _format_float(row["support_f1"], 4),
                    _format_float(row["coverage_at_budget"], 4),
                    _format_float(row["evidence_tokens"], 2),
                    _format_float(row["retrieval_latency_ms"], 4),
                    _format_float(row["packing_latency_ms"], 4),
                    _format_float(row["generation_latency_ms"], 2),
                    _format_float(row["total_non_generation_latency_ms"], 4),
                    _format_float(row["total_latency_ms"], 2),
                    _format_float(row["retrieval_pct_of_total"], 4),
                    _format_float(row["packing_pct_of_total"], 4),
                    _format_float(row["generation_pct_of_total"], 4),
                    _format_float(row["supportcover_overhead_vs_relevance_only_ms"], 4),
                    _format_float(row["peak_rss_mb"], 2),
                    str(row["experiment_id"]),
                    str(row["output_dir"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_analysis_markdown(summary_rows: list[dict[str, object]]) -> str:
    baseline = summary_rows[0]
    supportcover = summary_rows[1]
    answer_f1_gain = float(supportcover["answer_f1"]) - float(baseline["answer_f1"])
    support_f1_gain = float(supportcover["support_f1"]) - float(baseline["support_f1"])
    coverage_gain = float(supportcover["coverage_at_budget"]) - float(baseline["coverage_at_budget"])
    evidence_delta = float(supportcover["evidence_tokens"]) - float(baseline["evidence_tokens"])
    total_latency_delta = float(supportcover["total_latency_ms"]) - float(baseline["total_latency_ms"])

    lines = [
        "# Phase 8 Systems Analysis",
        "",
        "Phase 8 uses the frozen Qwen main comparison at token budget `160` and retrieval depth `5`, with `supportcover_final = no_redundancy`.",
        "",
        "| method | retrieval_pct_of_total | packing_pct_of_total | generation_pct_of_total | total_non_generation_latency_ms | total_latency_ms | peak_rss_mb |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    _format_float(row["retrieval_pct_of_total"], 4) + "%",
                    _format_float(row["packing_pct_of_total"], 4) + "%",
                    _format_float(row["generation_pct_of_total"], 4) + "%",
                    _format_float(row["total_non_generation_latency_ms"], 4),
                    _format_float(row["total_latency_ms"], 2),
                    _format_float(row["peak_rss_mb"], 2),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Main read:",
            f"- Retrieval accounts for about `{_format_float(baseline['retrieval_pct_of_total'], 4)}%` of total runtime for `relevance_only` and `{_format_float(supportcover['retrieval_pct_of_total'], 4)}%` for `supportcover_final`.",
            f"- Packing accounts for about `{_format_float(baseline['packing_pct_of_total'], 4)}%` of total runtime for `relevance_only` and `{_format_float(supportcover['packing_pct_of_total'], 4)}%` for `supportcover_final`.",
            f"- Generation dominates end-to-end cost: `{_format_float(baseline['generation_pct_of_total'], 4)}%` of total runtime for `relevance_only` and `{_format_float(supportcover['generation_pct_of_total'], 4)}%` for `supportcover_final`.",
            f"- `supportcover_final` adds only `{_format_float(supportcover['supportcover_overhead_vs_relevance_only_ms'], 4)}` ms of extra pre-generation latency over `relevance_only`, which is `{_format_float(supportcover['supportcover_overhead_vs_relevance_only_pct'], 2)}%` relative overhead on the small non-generation portion.",
            f"- The evidence-quality gains are achieved with modest systems cost: answer F1 improves by `{_format_float(answer_f1_gain, 4)}`, support F1 by `{_format_float(support_f1_gain, 4)}`, and coverage by `{_format_float(coverage_gain, 4)}`, while average evidence tokens change by only `{_format_float(evidence_delta, 2)}` and peak RSS changes by `{_format_float(supportcover['memory_delta_mb'], 2)}` MB.",
            f"- Total latency is not worse in the frozen run pair (`{_format_float(total_latency_delta, 2)}` ms delta for `supportcover_final` versus `relevance_only`) because generation dominates and varies more than the tiny packing overhead, which still fits the lightweight on-device framing.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_systems_summary(config: AppConfig) -> dict[str, Path]:
    summary_rows = _build_summary_rows(config)
    output_dir = ensure_dir(config.systems_summary.output_dir)
    summary_csv_path = output_dir / "phase8_systems_summary.csv"
    summary_md_path = output_dir / "phase8_systems_summary.md"
    analysis_path = output_dir / "phase8_systems_analysis.md"
    latency_breakdown_path = output_dir / config.systems_summary.figure_artifact_name

    write_csv(summary_csv_path, summary_rows)
    summary_md_path.write_text(
        _render_summary_markdown(config=config, summary_rows=summary_rows),
        encoding="utf-8",
    )
    analysis_path.write_text(_render_analysis_markdown(summary_rows), encoding="utf-8")
    write_csv(latency_breakdown_path, _build_latency_breakdown_rows(summary_rows))
    return {
        "summary_csv_path": summary_csv_path,
        "summary_md_path": summary_md_path,
        "analysis_path": analysis_path,
        "latency_breakdown_path": latency_breakdown_path,
    }
