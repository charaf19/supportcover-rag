from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from supportcover_rag.config import AppConfig, PathsConfig, SystemsSummaryConfig
from supportcover_rag.systems_summary import run_systems_summary


def _write_metrics(run_dir: Path, payload: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_systems_summary_writes_summary_and_breakdown(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    relevance_run = output_root / "robustness" / "EXP030_relevance_only_qwen_val_b160_d5_full"
    supportcover_run = output_root / "robustness" / "EXP031_supportcover_final_qwen_val_b160_d5_final"

    _write_metrics(
        relevance_run,
        {
            "experiment_id": "EXP030",
            "status": "completed",
            "method": "relevance_only",
            "variant": "full",
            "model": "qwen",
            "answer_f1": 0.40,
            "support_f1": 0.39,
            "coverage_at_budget": 0.58,
            "evidence_tokens": 155.0,
            "retrieval_latency_ms": 0.5,
            "packing_latency_ms": 2.0,
            "generation_latency_ms": 2000.0,
            "total_latency_ms": 2002.5,
            "peak_rss_mb": 1000.0,
            "num_examples": 100,
            "token_budget": 160,
            "retrieval_depth": 5,
            "output_dir": str(relevance_run),
        },
    )
    _write_metrics(
        supportcover_run,
        {
            "experiment_id": "EXP031",
            "status": "completed",
            "method": "supportcover_final",
            "variant": "final",
            "model": "qwen",
            "answer_f1": 0.43,
            "support_f1": 0.42,
            "coverage_at_budget": 0.65,
            "evidence_tokens": 154.5,
            "retrieval_latency_ms": 0.5,
            "packing_latency_ms": 2.2,
            "generation_latency_ms": 1990.0,
            "total_latency_ms": 1992.7,
            "peak_rss_mb": 1005.0,
            "num_examples": 100,
            "token_budget": 160,
            "retrieval_depth": 5,
            "output_dir": str(supportcover_run),
        },
    )

    config = AppConfig(
        paths=PathsConfig(output_root=str(output_root)),
        systems_summary=SystemsSummaryConfig(
            output_dir=str(output_root / "systems"),
            source_runs={
                "relevance_only": str(relevance_run),
                "supportcover_final": str(supportcover_run),
            },
        ),
    )

    artifacts = run_systems_summary(config)

    assert artifacts["summary_csv_path"].exists()
    assert artifacts["summary_md_path"].exists()
    assert artifacts["analysis_path"].exists()
    assert artifacts["latency_breakdown_path"].exists()

    with artifacts["summary_csv_path"].open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 2
    assert summary_rows[0]["method"] == "relevance_only"
    assert summary_rows[1]["method"] == "supportcover_final"
    assert float(summary_rows[1]["supportcover_overhead_vs_relevance_only_ms"]) == pytest.approx(0.2)
    assert float(summary_rows[1]["memory_delta_mb"]) == pytest.approx(5.0)

    with artifacts["latency_breakdown_path"].open("r", encoding="utf-8", newline="") as handle:
        breakdown_rows = list(csv.DictReader(handle))
    assert len(breakdown_rows) == 6
    assert {row["latency_component"] for row in breakdown_rows} == {"retrieval", "packing", "generation"}

    analysis_text = artifacts["analysis_path"].read_text(encoding="utf-8")
    assert "Phase 8 Systems Analysis" in analysis_text
    assert "supportcover_final" in analysis_text
