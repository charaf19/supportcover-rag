from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from supportcover_rag.io_utils import write_csv, write_json
from supportcover_rag.paper_artifacts import (
    DEVELOPMENT_SPLIT_SHA256,
    export_development_paper_results,
    export_protocol_paper_results,
)


def _packing_row(study: str, config_id: str, **overrides):
    return {
        "evaluation_scope": "packing_only",
        "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
        "study": study,
        "config_id": config_id,
        "factor": overrides.get("factor", ""),
        "value": overrides.get("value", ""),
        "mmr_lambda": overrides.get("mmr_lambda", ""),
        "variant": overrides.get("variant", ""),
        "num_examples": 2000,
        "support_em": 0.1,
        "support_precision": 0.2,
        "support_recall": 0.3,
        "support_f1": 0.24,
        "coverage_at_budget": 0.3,
        "evidence_tokens": 150.0,
    }


def _write_development_sources(tmp_path: Path) -> dict[str, Path]:
    packing_summary = tmp_path / "outputs" / "development" / "phase3" / "packing" / "packing_summary.csv"
    rows = [
        _packing_row(
            "supportcover_sensitivity",
            f"supportcover_beta_coverage_{index}",
            factor="beta_coverage",
            value=index,
        )
        for index in range(20)
    ]
    rows.extend(
        _packing_row("mmr_lambda", f"mmr_lambda_{value:g}", mmr_lambda=value)
        for value in (0.3, 0.5, 0.7, 0.9)
    )
    rows.extend(
        _packing_row("component_ablation", f"supportcover_{variant}", variant=variant)
        for variant in (
            "full",
            "no_query_coverage",
            "no_title_gain",
            "no_redundancy",
            "no_token_penalty",
        )
    )
    write_csv(packing_summary, rows)

    packing_manifest = packing_summary.parent / "packing_manifest.json"
    shortlist = tmp_path / "outputs" / "development" / "phase3" / "shortlist.json"
    decision = tmp_path / "outputs" / "development" / "phase3" / "decision.json"
    freeze_manifest = tmp_path / "configs" / "frozen" / "final_manifest.json"
    write_json(packing_manifest, {"status": "completed", "development_split_sha256": DEVELOPMENT_SPLIT_SHA256})
    write_json(
        shortlist,
        {
            "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
            "base_supportcover": {"config_id": "supportcover_beta_coverage_0"},
            "supportcover_candidates": [{"config_id": "supportcover_beta_coverage_1"}],
            "mmr_lambdas": [0.5],
        },
    )
    write_json(
        decision,
        {
            "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
            "selected_supportcover_coefficients": {
                "alpha_relevance": 1.0,
                "beta_coverage": 1.2,
                "gamma_redundancy": 0.6,
                "delta_token_cost": 0.15,
                "title_bonus": 0.3,
            },
            "selected_mmr_lambda_relevance": 0.5,
            "evidence_artifacts": {
                "packing_screen": "packing.csv",
                "generation_validation": "generation.csv",
                "mmr_selection": "mmr.csv",
                "component_ablation": "ablation.csv",
            },
        },
    )
    write_json(freeze_manifest, {"config_sha256": "a" * 64})
    return {
        "packing_summary": packing_summary,
        "packing_manifest": packing_manifest,
        "shortlist": shortlist,
        "decision": decision,
        "freeze_manifest": freeze_manifest,
    }


def test_development_export_is_curated_and_provenance_registered(tmp_path: Path) -> None:
    sources = _write_development_sources(tmp_path)
    output_root = tmp_path / "paper_results"
    artifacts = export_development_paper_results(
        packing_summary_path=sources["packing_summary"],
        packing_manifest_path=sources["packing_manifest"],
        shortlist_path=sources["shortlist"],
        decision_path=sources["decision"],
        freeze_manifest_path=sources["freeze_manifest"],
        output_root=output_root,
        code_revision="deadbeef",
    )

    assert set(artifacts) == {
        "sensitivity",
        "mmr_selection",
        "component_ablation",
        "development_decision",
    }
    with Path(artifacts["sensitivity"]).open(encoding="utf-8", newline="") as handle:
        sensitivity = list(csv.DictReader(handle))
    assert len(sensitivity) == 20
    assert {row["selection_status"] for row in sensitivity} >= {"base", "retained_for_generation"}
    assert not list(output_root.rglob("*.jsonl"))

    manifest_path = output_root / "08_reproducibility" / "paper_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 4
    assert all(entry["source_sha256"] for entry in manifest["artifacts"])
    assert all(entry["artifact_sha256"] for entry in manifest["artifacts"])
    assert all(entry["split_sha256"] == DEVELOPMENT_SPLIT_SHA256 for entry in manifest["artifacts"])


def test_development_export_refuses_incomplete_decision(tmp_path: Path) -> None:
    sources = _write_development_sources(tmp_path)
    write_json(sources["decision"], {"development_split_sha256": DEVELOPMENT_SPLIT_SHA256})
    with pytest.raises(ValueError, match="missing selected SupportCover coefficients"):
        export_development_paper_results(
            packing_summary_path=sources["packing_summary"],
            packing_manifest_path=sources["packing_manifest"],
            shortlist_path=sources["shortlist"],
            decision_path=sources["decision"],
            freeze_manifest_path=sources["freeze_manifest"],
            output_root=tmp_path / "paper_results",
        )


def test_protocol_export_requires_real_sources_and_registers_each_copy(tmp_path: Path) -> None:
    sources = _write_development_sources(tmp_path)
    split_validation = tmp_path / "data" / "splits" / "split_validation.json"
    frozen_config = tmp_path / "configs" / "final_frozen.yaml"
    environment = tmp_path / "outputs" / "development" / "phase3" / "environment.json"
    write_json(split_validation, {"status": "PASS"})
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_config.write_text("freeze:\n  sha256: " + "a" * 64 + "\n", encoding="utf-8")
    write_json(environment, {"accelerator": {"backend": "cuda"}})

    artifacts = export_protocol_paper_results(
        split_validation_path=split_validation,
        frozen_config_path=frozen_config,
        freeze_manifest_path=sources["freeze_manifest"],
        environment_path=environment,
        output_root=tmp_path / "paper_results",
    )
    assert all(Path(path).is_file() for path in artifacts.values())
    manifest = json.loads(
        (tmp_path / "paper_results" / "08_reproducibility" / "paper_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["artifacts"]) == 4
