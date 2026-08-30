from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from supportcover_rag.statistics_artifacts import run_statistics_plan


SPLIT_SHA = "a" * 64
METRICS = ["answer_em", "answer_f1", "support_f1", "support_recall", "coverage_at_budget"]


def _records(values: list[float], *, method: str) -> list[dict[str, Any]]:
    return [
        {
            "example_id": f"id-{index}",
            "method": method,
            "token_budget": 160,
            "answer_em": float(value == 1.0),
            "answer_f1": value,
            "support_f1": value,
            "support_recall": value,
            "coverage_at_budget": value,
        }
        for index, value in enumerate(values)
    ]


def _write_run(
    root: Path,
    *,
    run_id: str,
    method: str,
    values: list[float],
    role: str = "development",
    split_sha256: str = SPLIT_SHA,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir()
    prediction_rows = records if records is not None else _records(values, method=method)
    predictions = "".join(json.dumps(row) + "\n" for row in prediction_rows)
    (run_dir / "predictions.jsonl").write_text(predictions, encoding="utf-8")
    run_metadata = {
        "experiment_id": run_id,
        "method": method,
        "dataset": "hotpot_qa_distractor",
        "split": "train",
        "token_budget": 160,
        "retrieval_depth": 5,
        "split_sha256": split_sha256,
    }
    metrics = {
        **run_metadata,
        "status": "completed",
        "num_examples": len(values),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    config = {
        "raw_data": {"dataset_path": "hotpotqa/hotpot_qa", "dataset_config": "distractor"},
        "split": {"role": role},
        "experiments": {"split": "train"},
        "retrieval": {"method": "bm25", "bm25_k1": 1.5, "bm25_b": 0.75},
        "supportcover": {"token_budget": 160},
        "prompting": {"include_titles": True, "system_instruction": "fixed"},
        "generation": {
            "backend": "transformers",
            "model_name_or_path": "fixture/model",
            "temperature": 0.0,
            "max_new_tokens": 12,
            "do_sample": False,
        },
        "run": run_metadata,
    }
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_dir


def _write_plan(tmp_path: Path, reference: Path, comparison: Path) -> Path:
    plan = {
        "schema_version": 1,
        "project_root": ".",
        "dataset": "hotpot_qa_distractor",
        "split": "train",
        "role": "development",
        "split_sha256": SPLIT_SHA,
        "num_examples": 4,
        "comparison_family": "synthetic_fixture",
        "metrics": METRICS,
        "bootstrap_replicates": 100,
        "permutation_replicates": 100,
        "confidence_level": 0.95,
        "seed": 42,
        "holm_bonferroni": True,
        "reference": {"run_dir": str(reference), "method": "supportcover", "config_id": "reference"},
        "comparisons": [{"run_dir": str(comparison), "method": "mmr_sentence", "config_id": "comparison"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _run_fixture(plan: Path, output_dir: Path) -> dict[str, Any]:
    return run_statistics_plan(
        plan,
        output_dir=output_dir,
        allow_nonpublication_replicates=True,
        allow_unfrozen_for_tests=True,
    )


def test_same_ids_write_expected_statistics_artifacts(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(tmp_path, run_id="EXP002", method="mmr_sentence", values=[0.1, 0.3, 0.9, 1.0])
    plan = _write_plan(tmp_path, reference, comparison)

    artifacts = _run_fixture(plan, tmp_path / "statistics")

    assert set(artifacts) == {
        "method_summary",
        "paired_comparisons",
        "statistics_manifest",
        "statistics_manifest_sha256",
    }
    with Path(artifacts["method_summary"]).open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with Path(artifacts["paired_comparisons"]).open(encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    manifest = json.loads(Path(artifacts["statistics_manifest"]).read_text(encoding="utf-8"))
    assert len(summary_rows) == 10
    assert len(comparison_rows) == 5
    assert {row["test"] for row in comparison_rows} == {
        "mcnemar_exact_two_sided",
        "paired_random_sign_two_sided",
    }
    assert manifest["split_sha256"] == SPLIT_SHA
    assert manifest["num_examples"] == 4
    assert manifest["role"] == "development"
    assert manifest["artifacts"]["method_summary"]["sha256"]


def test_different_ids_fail_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    changed = _records([0.1, 0.3, 0.9, 1.0], method="mmr_sentence")
    changed[-1]["example_id"] = "different-id"
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        records=changed,
    )

    with pytest.raises(ValueError, match="populations do not match"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_missing_id_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    missing = _records([0.1, 0.3, 0.9], method="mmr_sentence")
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        records=missing,
    )

    with pytest.raises(ValueError, match="prediction count mismatch"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_duplicate_id_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    duplicate = _records([0.1, 0.3, 0.9, 1.0], method="mmr_sentence")
    duplicate[-1]["example_id"] = duplicate[0]["example_id"]
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        records=duplicate,
    )

    with pytest.raises(ValueError, match="Duplicate example ID"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_different_split_sha_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        split_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="split SHA256 mismatch"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_development_final_mixing_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        role="final",
    )

    with pytest.raises(ValueError, match="scientific-role mismatch"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_nan_metric_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    invalid = _records([0.1, 0.3, 0.9, 1.0], method="mmr_sentence")
    invalid[0]["answer_f1"] = float("nan")
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        records=invalid,
    )

    with pytest.raises(ValueError, match="must be finite"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_missing_metric_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    incomplete = _records([0.1, 0.3, 0.9, 1.0], method="mmr_sentence")
    del incomplete[0]["support_recall"]
    comparison = _write_run(
        tmp_path,
        run_id="EXP002",
        method="mmr_sentence",
        values=[0.1, 0.3, 0.9, 1.0],
        records=incomplete,
    )

    with pytest.raises(ValueError, match="missing metric 'support_recall'"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_protocol_mismatch_fails_closed(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(tmp_path, run_id="EXP002", method="mmr_sentence", values=[0.1, 0.3, 0.9, 1.0])
    config_path = comparison / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generation"]["max_new_tokens"] = 99
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation protocol"):
        _run_fixture(_write_plan(tmp_path, reference, comparison), tmp_path / "statistics")


def test_publication_runner_requires_phase3_freeze_manifest(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(tmp_path, run_id="EXP002", method="mmr_sentence", values=[0.1, 0.3, 0.9, 1.0])
    plan = _write_plan(tmp_path, reference, comparison)

    with pytest.raises(ValueError, match="phase3_freeze_manifest"):
        run_statistics_plan(plan, output_dir=tmp_path / "statistics")


def test_seeded_csv_outputs_are_deterministic(tmp_path: Path) -> None:
    reference = _write_run(tmp_path, run_id="EXP001", method="supportcover", values=[0.0, 0.2, 0.8, 1.0])
    comparison = _write_run(tmp_path, run_id="EXP002", method="mmr_sentence", values=[0.1, 0.3, 0.9, 1.0])
    plan = _write_plan(tmp_path, reference, comparison)

    first = _run_fixture(plan, tmp_path / "statistics-a")
    second = _run_fixture(plan, tmp_path / "statistics-b")

    assert Path(first["method_summary"]).read_bytes() == Path(second["method_summary"]).read_bytes()
    assert Path(first["paired_comparisons"]).read_bytes() == Path(second["paired_comparisons"]).read_bytes()
