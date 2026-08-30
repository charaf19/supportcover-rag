from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import typer
import yaml

from supportcover_rag.cli import _resolve_split
from supportcover_rag.config import (
    AppConfig,
    DevelopmentTuningConfig,
    ExperimentsConfig,
    PathsConfig,
    RuntimeConfig,
    SensitivityConfig,
    SplitConfig,
)
from supportcover_rag.generation import WhitespaceTokenCounter
from supportcover_rag.experiment_outputs import ExperimentFamily, ExperimentOutputManager
from supportcover_rag.io_utils import read_csv_rows, read_jsonl, write_csv, write_json, write_jsonl, write_yaml
from supportcover_rag.phase3 import (
    CANONICAL_COMPONENT_VARIANTS,
    aggregate_development_generation,
    build_packing_plan,
    freeze_development_selection,
    run_packing_screen,
    validate_development_protocol,
)
from supportcover_rag.splits import build_split_manifest, ordered_ids_sha256


def _write_population(tmp_path, ids: list[str]):
    data_root = tmp_path / "data"
    processed = data_root / "processed" / "train.jsonl"
    rows = [
        {
            "id": item_id,
            "question": "Where is Alpha located?",
            "answer": "Beta",
            "type": "bridge",
            "level": "easy",
            "context": [
                {"title": "Alpha", "sentences": ["Alpha is located in Beta.", "Unrelated sentence."]},
                {"title": "Beta", "sentences": ["Beta is a place."]},
            ],
            "supporting_facts": [{"title": "Alpha", "sent_id": 0}],
        }
        for item_id in ids
    ]
    write_jsonl(processed, rows)
    ids_path = data_root / "splits" / "development_ids.json"
    write_json(
        ids_path,
        build_split_manifest(
            ids,
            ids_file=ids_path,
            role="development",
            seed=42,
            stratify_by=["type", "level"],
            source_path=processed,
        ),
    )
    return data_root, ids_path


def _config(tmp_path, ids: list[str]) -> AppConfig:
    data_root, ids_path = _write_population(tmp_path, ids)
    return AppConfig(
        paths=PathsConfig(data_root=str(data_root), output_root=str(tmp_path / "outputs")),
        split=SplitConfig(ids_file=str(ids_path), role="development", stratify_by=["type", "level"]),
        experiments=ExperimentsConfig(split="train", methods=["supportcover"]),
        development_tuning=DevelopmentTuningConfig(
            output_dir=str(tmp_path / "outputs" / "development" / "phase3"),
            expected_development_count=len(ids),
            expected_development_sha256=ordered_ids_sha256(ids),
            expected_final_count=1,
            expected_final_sha256=ordered_ids_sha256(["final-1"]),
        ),
    )


def test_phase3_plan_is_ofat_and_excludes_legacy_ablation(tmp_path):
    config = _config(tmp_path, ["dev-1"])
    plan = build_packing_plan(config)

    assert len(plan) == 29
    assert [item.mmr_lambda for item in plan if item.study == "mmr_lambda"] == [0.3, 0.5, 0.7, 0.9]
    assert tuple(item.variant for item in plan if item.study == "component_ablation") == CANONICAL_COMPONENT_VARIANTS
    assert all(item.variant != "no_coverage" for item in plan)

    base = config.supportcover
    for item in (entry for entry in plan if entry.study == "supportcover_sensitivity"):
        changed = {
            field
            for field in ("beta_coverage", "title_bonus", "delta_token_cost", "gamma_redundancy")
            if getattr(item.supportcover, field) != getattr(base, field)
        }
        assert changed <= {item.factor}


def test_phase3_plan_rejects_a_changed_preregistered_grid(tmp_path):
    config = _config(tmp_path, ["dev-1"])
    config = replace(config, sensitivity=SensitivityConfig(beta=[0.3, 0.6]))
    with pytest.raises(ValueError, match="beta grid must be exactly"):
        build_packing_plan(config)


@pytest.mark.parametrize("role,split", [("final", "train"), ("development", "validation"), ("", "train")])
def test_phase3_protocol_rejects_non_development_train_inputs(tmp_path, role, split):
    config = _config(tmp_path, ["dev-1"])
    config = replace(config, split=replace(config.split, role=role), experiments=replace(config.experiments, split=split))
    with pytest.raises(ValueError):
        validate_development_protocol(config)


def test_development_cli_rejects_validation_before_loading_it(tmp_path):
    config = _config(tmp_path, ["dev-1"])
    with pytest.raises(typer.BadParameter, match="locked to the processed train split"):
        _resolve_split("validation", config)


def test_packing_screen_is_generator_free_and_records_all_rows(tmp_path):
    config = _config(tmp_path, ["dev-1", "dev-2"])
    manifest = run_packing_screen(config, token_counter=WhitespaceTokenCounter())

    assert manifest["evaluation_scope"] == "packing_only_no_generator"
    assert manifest["development_count"] == 2
    assert manifest["num_plan_items"] == 29
    assert manifest["num_metric_rows"] == 58
    packing_dir = tmp_path / "outputs" / "development" / "phase3" / "packing"
    assert len(read_jsonl(packing_dir / "supportcover_sensitivity.jsonl")) == 40
    assert len(read_jsonl(packing_dir / "mmr_lambda.jsonl")) == 8
    assert len(read_jsonl(packing_dir / "component_ablation.jsonl")) == 10


def test_freeze_requires_development_evidence_and_is_deterministic(tmp_path):
    config = _config(tmp_path, ["dev-1", "dev-2"])
    evidence_path = tmp_path / "outputs" / "development" / "phase3" / "development_results.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("metric,value\nsupport_f1,1.0\n", encoding="utf-8")
    final_ids_path = tmp_path / "data" / "splits" / "final_ids.json"
    write_json(
        final_ids_path,
        build_split_manifest(
            ["final-1"],
            ids_file=final_ids_path,
            role="final",
            seed=42,
            source_path=tmp_path / "data" / "processed" / "validation.jsonl",
        ),
    )
    decision_path = tmp_path / "decision.json"
    write_json(
        decision_path,
        {
            "development_split_sha256": config.development_tuning.expected_development_sha256,
            "selected_supportcover_coefficients": {
                "alpha_relevance": 1.0,
                "beta_coverage": 1.2,
                "gamma_redundancy": 0.6,
                "delta_token_cost": 0.15,
                "title_bonus": 0.3,
            },
            "selected_mmr_lambda_relevance": 0.5,
            "evidence_artifacts": {
                "packing_screen": str(evidence_path),
                "generation_validation": str(evidence_path),
                "mmr_selection": str(evidence_path),
                "component_ablation": str(evidence_path),
            },
            "decision_notes": "Synthetic unit-test decision.",
        },
    )

    manifest_one = freeze_development_selection(
        config,
        decision_path=decision_path,
        final_ids_path=final_ids_path,
        final_config_path=tmp_path / "final_one.yaml",
        manifest_path=tmp_path / "manifest_one.json",
    )
    manifest_two = freeze_development_selection(
        config,
        decision_path=decision_path,
        final_ids_path=final_ids_path,
        final_config_path=tmp_path / "final_two.yaml",
        manifest_path=tmp_path / "manifest_two.json",
    )

    assert manifest_one == manifest_two
    assert manifest_one["final_predictions_inspected"] is False
    assert manifest_one["configuration"]["development_split_sha256"] == ordered_ids_sha256(["dev-1", "dev-2"])
    assert manifest_one["configuration"]["final_split_sha256"] == ordered_ids_sha256(["final-1"])
    frozen = yaml.safe_load((tmp_path / "final_one.yaml").read_text(encoding="utf-8"))
    assert frozen["split"]["role"] == "final"
    assert frozen["experiments"]["split"] == "validation"
    assert frozen["freeze"]["sha256"] == manifest_one["config_sha256"]


def test_checked_in_phase3_config_uses_train_and_frozen_development_manifest():
    payload = yaml.safe_load(open("configs/phase3_sensitivity.yaml", encoding="utf-8"))
    assert payload["experiments"]["split"] == "train"
    assert payload["raw_data"]["splits"] == ["train"]
    assert payload["split"] == {
        "ids_file": "./data/splits/development_ids.json",
        "role": "development",
        "stratify_by": ["type", "level"],
    }
    assert "no_coverage" not in payload["ablations"]["variants"]


def test_phase3_templates_use_canonical_development_sha256():
    from supportcover_rag.paper_artifacts import DEVELOPMENT_SPLIT_SHA256

    template_paths = (
        "configs/phase3_shortlist.template.json",
        "configs/phase3_decision.template.json",
    )
    for template_path in template_paths:
        with open(template_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["development_split_sha256"] == DEVELOPMENT_SPLIT_SHA256


def test_generation_aggregation_validates_shortlist_and_run_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected_sha = "0e02afdcdff360d26725abe9c197a457dcbe76c92aa54338cdc146806b9ed7c6"
    base_coefficients = {
        "alpha_relevance": 1.0,
        "beta_coverage": 1.2,
        "gamma_redundancy": 0.6,
        "delta_token_cost": 0.15,
        "title_bonus": 0.3,
    }
    run_specs = (
        ("supportcover_beta_coverage_1.2", "supportcover", "EXP001", base_coefficients, 0.5),
        ("mmr_lambda_0.5", "mmr_sentence", "EXP002", base_coefficients, 0.5),
    )
    generation_plan = []
    for config_id, method, experiment_id, coefficients, mmr_lambda in run_specs:
        run_dir = Path("outputs/development/phase3/runs/baseline") / f"{experiment_id}_{method}"
        write_csv(
            run_dir / "summary.csv",
            [
                {
                    "experiment_id": experiment_id,
                    "status": "completed",
                    "method": method,
                    "split": "train",
                    "split_sha256": expected_sha,
                    "num_examples": 2000,
                    "token_budget": 160,
                    "retrieval_depth": 5,
                    "answer_em": 0.1,
                    "answer_f1": 0.2,
                    "support_f1": 0.3,
                    "support_recall": 0.4,
                    "coverage_at_budget": 0.4,
                    "evidence_tokens": 150.0,
                }
            ],
        )
        write_jsonl(run_dir / "predictions.jsonl", [{"example_id": "synthetic"}])
        write_yaml(
            run_dir / "config.resolved.yaml",
            {
                "supportcover": coefficients,
                "retrieval": {"mmr_lambda_relevance": mmr_lambda},
            },
        )
        generation_plan.append(
            {
                "config_id": config_id,
                "method": method,
                "experiment_id": experiment_id,
                "run_dir": str(run_dir),
            }
        )

    shortlist_path = Path("outputs/development/phase3/shortlist.json")
    write_json(
        shortlist_path,
        {
            "development_split_sha256": expected_sha,
            "base_supportcover": {
                "config_id": "supportcover_beta_coverage_1.2",
                "coefficients": base_coefficients,
            },
            "supportcover_candidates": [],
            "mmr_lambdas": [0.5],
            "generation_plan": generation_plan,
        },
    )
    output_path = Path("outputs/development/phase3/generation_validation.csv")
    rows = aggregate_development_generation(shortlist_path=shortlist_path, output_path=output_path)

    assert [row["config_id"] for row in rows] == ["supportcover_beta_coverage_1.2", "mmr_lambda_0.5"]
    assert len(read_csv_rows(output_path)) == 2
    assert all(row["source_predictions_sha256"] for row in rows)


def test_explicit_experiment_id_can_resume_without_duplicate_registry_rows(tmp_path):
    config = AppConfig(
        paths=PathsConfig(output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(resume=True, overwrite=False),
    )
    manager = ExperimentOutputManager(config.paths.output_root)
    context = manager.prepare_run(
        config=config,
        family=ExperimentFamily.BASELINE,
        method="supportcover",
        split_name="train",
        token_budget=160,
        retrieval_depth=5,
        variant="full",
        experiment_id="EXP001",
        split_sha256=config.development_tuning.expected_development_sha256,
    )
    context.output_dir.mkdir(parents=True)
    manager.write_config_snapshot(context.output_dir / "config.resolved.yaml", config, context)
    manager.append_registry_row({"experiment_id": "EXP001", "status": "failed"})

    resumed = manager.prepare_run(
        config=config,
        family=ExperimentFamily.BASELINE,
        method="supportcover",
        split_name="train",
        token_budget=160,
        retrieval_depth=5,
        variant="full",
        experiment_id="EXP001",
        split_sha256=config.development_tuning.expected_development_sha256,
    )
    manager.append_registry_row({"experiment_id": "EXP001", "status": "completed"})

    assert resumed.output_dir == context.output_dir
    assert resumed.timestamp == context.timestamp
    registry = read_csv_rows(manager.registry_path)
    assert len(registry) == 1
    assert registry[0]["status"] == "completed"
