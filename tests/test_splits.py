from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from supportcover_rag.cli import app
from supportcover_rag.config import (
    AppConfig,
    ExperimentsConfig,
    GenerationConfig,
    PathsConfig,
    RuntimeConfig,
    SplitConfig,
    load_config,
)
from supportcover_rag.data import load_examples_by_ids
from supportcover_rag.experiment_outputs import ExperimentFamily
from supportcover_rag.pipeline import ExperimentRunner
from supportcover_rag.splits import (
    build_record_strata,
    build_split_manifest,
    load_json_ids,
    ordered_ids_sha256,
    select_seeded_stratified_ids,
    validate_disjoint_splits,
    validate_unique_ids,
)


def _processed_row(example_id: str) -> dict[str, object]:
    return {
        "id": example_id,
        "question": f"Question {example_id}?",
        "answer": example_id,
        "type": "bridge",
        "level": "easy",
        "context": [{"title": "Title", "sentences": ["Evidence."]}],
        "supporting_facts": [{"title": "Title", "sent_id": 0}],
    }


def _write_processed(path: Path, ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps(_processed_row(example_id)) + "\n" for example_id in ids),
        encoding="utf-8",
    )


def test_seeded_stratified_selection_is_deterministic() -> None:
    ids = [f"id-{index}" for index in range(12)]
    strata = {item_id: ("bridge" if index < 8 else "comparison") for index, item_id in enumerate(ids)}

    first = select_seeded_stratified_ids(ids, sample_size=6, seed=17, strata=strata)
    second = select_seeded_stratified_ids(ids, sample_size=6, seed=17, strata=strata)

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6


def test_seed_changes_deterministic_sample() -> None:
    ids = [f"id-{index}" for index in range(20)]

    assert select_seeded_stratified_ids(ids, sample_size=7, seed=1) != select_seeded_stratified_ids(
        ids,
        sample_size=7,
        seed=2,
    )


def test_combined_strata_and_manifest_are_self_contained(tmp_path: Path) -> None:
    rows = [
        {**_processed_row("a"), "type": "bridge", "level": "easy"},
        {**_processed_row("b"), "type": "comparison", "level": "hard"},
    ]
    strata = build_record_strata(rows, ["type", "level"])
    manifest = build_split_manifest(
        ["b", "a"],
        ids_file=tmp_path / "development_ids.json",
        role="Development",
        seed=42,
        stratify_by=["type", "level"],
        source_path=tmp_path / "train.jsonl",
    )

    assert strata["a"] != strata["b"]
    assert manifest["role"] == "development"
    assert manifest["ids"] == ["b", "a"]
    assert manifest["stratify_by"] == ["type", "level"]
    assert manifest["split_sha256"] == ordered_ids_sha256(["b", "a"])


def test_split_disjointness_accepts_zero_overlap_and_identifies_overlap() -> None:
    validate_disjoint_splits({"development": ["a", "b"], "final": ["c", "d"]})

    with pytest.raises(ValueError, match="shared"):
        validate_disjoint_splits({"development": ["a", "shared"], "final": ["shared", "d"]})


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_unique_ids(["a", "b", "a"])


def test_ordered_id_sha256_is_stable_and_order_sensitive() -> None:
    ids = ["a", "b", "c"]

    assert ordered_ids_sha256(ids) == ordered_ids_sha256(list(ids))
    assert ordered_ids_sha256(ids) != ordered_ids_sha256(list(reversed(ids)))


def test_load_examples_by_ids_preserves_requested_order(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.jsonl"
    _write_processed(processed_path, ["a", "b", "c"])

    examples = load_examples_by_ids(processed_path, ["c", "a"])

    assert [example.example_id for example in examples] == ["c", "a"]


def test_load_examples_by_ids_reports_missing_ids(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.jsonl"
    _write_processed(processed_path, ["a", "b"])

    with pytest.raises(ValueError, match="missing"):
        load_examples_by_ids(processed_path, ["a", "missing"])


def test_split_config_normalizes_legacy_scalar_stratification(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"split": {"ids_file": "ids.json", "role": "development", "stratify_by": "type"}}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.split.stratify_by == ["type"]


def test_explicit_ids_override_runtime_limit_and_preserve_order(tmp_path: Path) -> None:
    split_path = tmp_path / "validation.jsonl"
    _write_processed(split_path, ["a", "b", "c"])
    ids_path = tmp_path / "development_ids.json"
    ids_path.write_text(json.dumps(["c", "a"]), encoding="utf-8")
    config = AppConfig(
        paths=PathsConfig(output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1),
        split=SplitConfig(ids_file=str(ids_path), role="development"),
        generation=GenerationConfig(backend="echo", batch_size=1),
        experiments=ExperimentsConfig(methods=["relevance_only"]),
    )
    runner = ExperimentRunner(config)

    result = runner.run_single(
        split_path=split_path,
        split_name="validation",
        method="relevance_only",
        token_budget=160,
        retrieval_depth=5,
        family=ExperimentFamily.DEBUG,
    )

    assert result["num_examples"] == 2
    predictions = [json.loads(line) for line in (Path(str(result["output_dir"])) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["example_id"] for record in predictions] == ["c", "a"]
    assert result["split_sha256"] == ordered_ids_sha256(["c", "a"])


def test_final_split_rejects_runtime_limit_even_with_explicit_ids(tmp_path: Path) -> None:
    split_path = tmp_path / "validation.jsonl"
    _write_processed(split_path, ["a"])
    ids_path = tmp_path / "final_ids.json"
    ids_path.write_text(json.dumps(["a"]), encoding="utf-8")
    config = AppConfig(
        paths=PathsConfig(output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=1),
        split=SplitConfig(ids_file=str(ids_path), role="final"),
        generation=GenerationConfig(backend="echo", batch_size=1),
    )
    runner = ExperimentRunner(config)

    with pytest.raises(ValueError, match="runtime.limit must be null"):
        runner.run_single(
            split_path=split_path,
            split_name="validation",
            method="relevance_only",
            token_budget=160,
            retrieval_depth=5,
            family=ExperimentFamily.DEBUG,
        )


def test_split_cli_creates_and_validates_disjoint_manifests(tmp_path: Path) -> None:
    development_source = tmp_path / "train.jsonl"
    final_source = tmp_path / "validation.jsonl"
    _write_processed(development_source, ["train-a", "train-b", "train-c"])
    _write_processed(final_source, ["val-a", "val-b"])
    development_manifest = tmp_path / "development_ids.json"
    final_manifest = tmp_path / "final_ids.json"
    report_path = tmp_path / "split_validation.json"
    cli = CliRunner()

    development_result = cli.invoke(
        app,
        [
            "create-split",
            "--processed",
            str(development_source),
            "--output",
            str(development_manifest),
            "--role",
            "development",
            "--sample-size",
            "2",
            "--stratify-by",
            "type,level",
        ],
    )
    final_result = cli.invoke(
        app,
        [
            "create-split",
            "--processed",
            str(final_source),
            "--output",
            str(final_manifest),
            "--role",
            "final",
        ],
    )
    validation_result = cli.invoke(
        app,
        [
            "validate-splits",
            "--development",
            str(development_manifest),
            "--final",
            str(final_manifest),
            "--output",
            str(report_path),
        ],
    )

    assert development_result.exit_code == 0, development_result.output
    assert final_result.exit_code == 0, final_result.output
    assert validation_result.exit_code == 0, validation_result.output
    assert len(load_json_ids(development_manifest)) == 2
    assert load_json_ids(final_manifest) == ["val-a", "val-b"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["overlap_count"] == 0
