from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from supportcover_rag.config import AblationsConfig, AppConfig, ExperimentsConfig, GenerationConfig, PathsConfig, RobustnessConfig, RuntimeConfig
from supportcover_rag.experiment_outputs import (
    ExperimentFamily,
    build_run_folder_name,
    resolve_model_alias,
    resolve_split_alias,
    validate_experiment_id,
)
from supportcover_rag.logging_utils import configure_logging
from supportcover_rag.pipeline import ExperimentRunner


def test_output_naming_helpers_use_short_aliases() -> None:
    assert resolve_model_alias("Qwen/Qwen3-4B-Instruct-2507") == "qwen"
    assert resolve_model_alias("google/gemma-2-2b-it") == "gemma"
    assert resolve_split_alias("validation") == "val"
    assert (
        build_run_folder_name(
            experiment_id="EXP031",
            method="supportcover",
            model_alias="qwen",
            split="val",
            token_budget=160,
            retrieval_depth=5,
            variant="no_coverage",
        )
        == "EXP031_supportcover_qwen_val_b160_d5_no_coverage"
    )


def test_validate_experiment_id_enforces_debug_prefix() -> None:
    with pytest.raises(ValueError, match="Debug runs must use DBG ids"):
        validate_experiment_id("EXP001", ExperimentFamily.DEBUG)

    with pytest.raises(ValueError, match="Paper-grade runs must use EXP ids"):
        validate_experiment_id("DBG001", ExperimentFamily.BASELINE)


def test_optional_provenance_is_persisted_without_changing_legacy_payloads(tmp_path: Path) -> None:
    config = AppConfig(
        paths=PathsConfig(output_root=str(tmp_path / "outputs")),
        generation=GenerationConfig(backend="echo"),
    )
    runner = ExperimentRunner(config)
    context = runner.output_manager.prepare_run(
        config=config,
        family=ExperimentFamily.DEBUG,
        method="relevance_only",
        split_name="validation",
        token_budget=160,
        retrieval_depth=5,
        variant="full",
        config_sha256="a" * 64,
        code_revision="79e2d4c1e3233c94a9a0faf80be770596a0bc72b",
        split_sha256="b" * 64,
    )

    payload = runner._build_run_payload(context, {"num_examples": 1}, status="completed", notes="")
    assert payload["config_sha256"] == "a" * 64
    assert payload["code_revision"] == "79e2d4c1e3233c94a9a0faf80be770596a0bc72b"
    assert payload["split_sha256"] == "b" * 64

    legacy_context = runner.output_manager.prepare_run(
        config=config,
        family=ExperimentFamily.DEBUG,
        method="supportcover",
        split_name="validation",
        token_budget=160,
        retrieval_depth=5,
        variant="full",
    )
    legacy_payload = runner._build_run_payload(
        legacy_context,
        {"num_examples": 1},
        status="completed",
        notes="",
    )
    assert "config_sha256" not in legacy_payload
    assert "code_revision" not in legacy_payload
    assert "split_sha256" not in legacy_payload


def test_runner_writes_new_output_layout_and_registry(tmp_path: Path) -> None:
    configure_logging("INFO")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "validation.jsonl"
    example = {
        "id": "ex-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "type": "bridge",
        "level": "easy",
        "context": [
            {"title": "Hamlet", "sentences": ["Hamlet is a tragedy written by William Shakespeare."]},
            {"title": "Macbeth", "sentences": ["Macbeth is another play by Shakespeare."]},
        ],
        "supporting_facts": [{"title": "Hamlet", "sent_id": 0}],
    }
    split_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1, overwrite=False),
        generation=GenerationConfig(
            backend="echo",
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=1,
        ),
        experiments=ExperimentsConfig(methods=["relevance_only"]),
    )
    runner = ExperimentRunner(config)

    baseline = runner.run_single(
        split_path=split_path,
        split_name="validation",
        method="relevance_only",
        token_budget=160,
        retrieval_depth=5,
        family=ExperimentFamily.BASELINE,
        notes="baseline smoke",
    )
    debug = runner.run_single(
        split_path=split_path,
        split_name="validation",
        method="supportcover",
        token_budget=160,
        retrieval_depth=5,
        family=ExperimentFamily.DEBUG,
        notes="debug smoke",
    )

    baseline_dir = Path(str(baseline["output_dir"]))
    debug_dir = Path(str(debug["output_dir"]))

    assert baseline["experiment_id"] == "EXP001"
    assert baseline_dir.name == "EXP001_relevance_only_qwen_val_b160_d5_full"
    assert debug["experiment_id"] == "DBG001"
    assert debug_dir.name == "DBG001_supportcover_qwen_val_b160_d5_full"

    for run_dir in (baseline_dir, debug_dir):
        assert (run_dir / "config.resolved.yaml").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "run.log").exists()

    resolved_config = yaml.safe_load((baseline_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config["run"]["experiment_id"] == "EXP001"
    assert resolved_config["run"]["family"] == "baseline"
    assert resolved_config["run"]["model_alias"] == "qwen"

    registry_path = tmp_path / "outputs" / "registry" / "experiments.csv"
    latest_path = tmp_path / "outputs" / "registry" / "latest.json"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "EXP001"
    assert rows[0]["family"] == "baseline"
    assert rows[0]["status"] == "completed"
    assert rows[0]["method"] == "relevance_only"
    assert rows[0]["model"] == "qwen"
    assert rows[0]["split"] == "val"
    assert rows[0]["notes"] == "baseline smoke"
    assert rows[1]["experiment_id"] == "DBG001"
    assert rows[1]["family"] == "debug"

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["experiment_id"] == "DBG001"
    assert latest["family"] == "debug"


def test_budget_ablation_uses_configured_methods_and_writes_summary(tmp_path: Path) -> None:
    configure_logging("INFO")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "validation.jsonl"
    example = {
        "id": "ex-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "type": "bridge",
        "level": "easy",
        "context": [
            {"title": "Hamlet", "sentences": ["Hamlet is a tragedy written by William Shakespeare."]},
            {"title": "Macbeth", "sentences": ["Macbeth is another play by Shakespeare."]},
        ],
        "supporting_facts": [{"title": "Hamlet", "sent_id": 0}],
    }
    split_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1, overwrite=False),
        generation=GenerationConfig(
            backend="echo",
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=1,
        ),
        experiments=ExperimentsConfig(methods=["relevance_only", "supportcover"]),
        ablations=AblationsConfig(token_budgets=[96, 128], retrieval_depths=[5], variants=["full"]),
    )
    runner = ExperimentRunner(config)

    results = runner.run_ablations(
        split_path=split_path,
        split_name="validation",
        family=ExperimentFamily.ABLATION_BUDGET,
        notes="budget smoke",
    )

    assert len(results) == 4
    assert [(row["method"], row["token_budget"]) for row in results] == [
        ("relevance_only", 96),
        ("supportcover", 96),
        ("relevance_only", 128),
        ("supportcover", 128),
    ]

    summary_path = tmp_path / "outputs" / "ablation_budget" / "EXP001_EXP004_comparison.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["method"] == "relevance_only"
    assert rows[1]["method"] == "supportcover"


def test_depth_ablation_uses_configured_methods_and_writes_summary(tmp_path: Path) -> None:
    configure_logging("INFO")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "validation.jsonl"
    example = {
        "id": "ex-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "type": "bridge",
        "level": "easy",
        "context": [
            {"title": "Hamlet", "sentences": ["Hamlet is a tragedy written by William Shakespeare."]},
            {"title": "Macbeth", "sentences": ["Macbeth is another play by Shakespeare."]},
        ],
        "supporting_facts": [{"title": "Hamlet", "sent_id": 0}],
    }
    split_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1, overwrite=False),
        generation=GenerationConfig(
            backend="echo",
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=1,
        ),
        experiments=ExperimentsConfig(methods=["relevance_only", "supportcover"]),
        ablations=AblationsConfig(token_budgets=[160], retrieval_depths=[5, 10], variants=["full"]),
    )
    runner = ExperimentRunner(config)

    results = runner.run_ablations(
        split_path=split_path,
        split_name="validation",
        family=ExperimentFamily.ABLATION_DEPTH,
        notes="depth smoke",
    )

    assert len(results) == 4
    assert [(row["method"], row["retrieval_depth"]) for row in results] == [
        ("relevance_only", 5),
        ("supportcover", 5),
        ("relevance_only", 10),
        ("supportcover", 10),
    ]

    summary_path = tmp_path / "outputs" / "ablation_depth" / "EXP001_EXP004_comparison.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["method"] == "relevance_only"
    assert rows[1]["method"] == "supportcover"


def test_component_ablation_uses_configured_variants_and_writes_summary(tmp_path: Path) -> None:
    configure_logging("INFO")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "validation.jsonl"
    example = {
        "id": "ex-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "type": "bridge",
        "level": "easy",
        "context": [
            {"title": "Hamlet", "sentences": ["Hamlet is a tragedy written by William Shakespeare."]},
            {"title": "Macbeth", "sentences": ["Macbeth is another play by Shakespeare."]},
        ],
        "supporting_facts": [{"title": "Hamlet", "sent_id": 0}],
    }
    split_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1, overwrite=False),
        generation=GenerationConfig(
            backend="echo",
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=1,
        ),
        experiments=ExperimentsConfig(methods=["relevance_only", "supportcover"]),
        ablations=AblationsConfig(token_budgets=[160], retrieval_depths=[5], variants=["relevance_only", "full", "no_coverage"]),
    )
    runner = ExperimentRunner(config)

    results = runner.run_ablations(
        split_path=split_path,
        split_name="validation",
        family=ExperimentFamily.ABLATION_COMPONENT,
        notes="component smoke",
    )

    assert len(results) == 3
    assert [(row["method"], row["variant"]) for row in results] == [
        ("relevance_only", "relevance_only"),
        ("supportcover", "full"),
        ("supportcover", "no_coverage"),
    ]

    summary_path = tmp_path / "outputs" / "ablation_component" / "EXP001_EXP003_comparison.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["variant"] == "relevance_only"
    assert rows[1]["variant"] == "full"


def test_robustness_suite_uses_canonical_supportcover_variant_and_writes_summary(tmp_path: Path) -> None:
    configure_logging("INFO")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "validation.jsonl"
    example = {
        "id": "ex-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "type": "bridge",
        "level": "easy",
        "context": [
            {"title": "Hamlet", "sentences": ["Hamlet is a tragedy written by William Shakespeare."]},
            {"title": "Macbeth", "sentences": ["Macbeth is another play by Shakespeare."]},
        ],
        "supporting_facts": [{"title": "Hamlet", "sent_id": 0}],
    }
    split_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1, overwrite=False),
        generation=GenerationConfig(
            backend="echo",
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=1,
        ),
        robustness=RobustnessConfig(
            models=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen3-4B-Instruct-2507"],
            supportcover_final_variant="no_redundancy",
        ),
    )
    runner = ExperimentRunner(config)

    results = runner.run_robustness_suite(
        split_path=split_path,
        split_name="validation",
        notes="robustness smoke",
    )

    assert len(results) == 4
    assert [(row["model"], row["method"], row["variant"]) for row in results] == [
        ("tinyllama", "relevance_only", "full"),
        ("tinyllama", "supportcover_final", "final"),
        ("qwen", "relevance_only", "full"),
        ("qwen", "supportcover_final", "final"),
    ]

    summary_path = tmp_path / "outputs" / "robustness" / "EXP001_EXP004_comparison.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[1]["method"] == "supportcover_final"
    assert rows[1]["variant"] == "final"
