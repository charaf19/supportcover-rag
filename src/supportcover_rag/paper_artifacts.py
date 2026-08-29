from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from supportcover_rag.io_utils import ensure_dir, write_csv, write_json


PAPER_MANIFEST_SCHEMA_VERSION = 1
DEVELOPMENT_SPLIT_SHA256 = "0e02afdcdff360d26725abe9c197a457dcbe76c92aa54338cdc146806b9ed7c6"
FINAL_SPLIT_SHA256 = "fc5c4bbd3b2a0304803f118cc098eec9d78521ac7f769877774239f52a4ecf6c"
PACKING_COLUMNS = (
    "evaluation_scope",
    "study",
    "config_id",
    "factor",
    "value",
    "mmr_lambda",
    "variant",
    "num_examples",
    "support_em",
    "support_precision",
    "support_recall",
    "support_f1",
    "coverage_at_budget",
    "evidence_tokens",
    "selection_status",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _relative_or_resolved(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _manifest_path(output_root: str | Path) -> Path:
    return Path(output_root) / "08_reproducibility" / "paper_artifact_manifest.json"


def register_paper_artifact(
    *,
    artifact_path: str | Path,
    role: str,
    source_artifacts: Sequence[str | Path],
    output_root: str | Path = "paper_results",
    split_sha256: str | Mapping[str, str] | None = None,
    config_sha256: str | None = None,
    freeze_sha256: str | None = None,
    code_revision: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Upsert provenance for one real, already-written publication artifact."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Publication artifact does not exist: {artifact}")
    sources = [Path(path) for path in source_artifacts]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Publication artifact sources do not exist: " + ", ".join(missing))

    manifest_path = _manifest_path(output_root)
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path)
        if manifest.get("schema_version") != PAPER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported paper artifact manifest schema version.")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise ValueError("Paper artifact manifest must contain an artifacts list.")
    else:
        manifest = {"schema_version": PAPER_MANIFEST_SCHEMA_VERSION, "artifacts": []}
        entries = manifest["artifacts"]

    artifact_name = _relative_or_resolved(artifact)
    source_names = [_relative_or_resolved(path) for path in sources]
    entry = {
        "artifact": artifact_name,
        "role": role,
        "source_artifacts": source_names,
        "source_sha256": [file_sha256(path) for path in sources],
        "artifact_sha256": file_sha256(artifact),
        "split_sha256": split_sha256,
        "config_sha256": config_sha256,
        "freeze_sha256": freeze_sha256,
        "code_revision": code_revision,
        "generated_at": generated_at or _utc_timestamp(),
    }
    manifest["artifacts"] = sorted(
        [existing for existing in entries if existing.get("artifact") != artifact_name] + [entry],
        key=lambda item: str(item["artifact"]),
    )
    manifest["generated_at"] = generated_at or _utc_timestamp()
    write_json(manifest_path, manifest)
    return entry


def _require_completed_decision(decision: Mapping[str, Any]) -> None:
    if decision.get("development_split_sha256") != DEVELOPMENT_SPLIT_SHA256:
        raise ValueError("Development decision does not match the frozen development split SHA256.")
    if not isinstance(decision.get("selected_supportcover_coefficients"), dict):
        raise ValueError("Development decision is missing selected SupportCover coefficients.")
    if not isinstance(decision.get("selected_mmr_lambda_relevance"), (int, float)):
        raise ValueError("Development decision is missing the selected MMR lambda.")
    evidence = decision.get("evidence_artifacts")
    if not isinstance(evidence, dict) or not all(evidence.get(name) for name in (
        "packing_screen",
        "generation_validation",
        "mmr_selection",
        "component_ablation",
    )):
        raise ValueError("Development decision is incomplete: all evidence artifacts are required.")


def _shortlisted_config_ids(shortlist: Mapping[str, Any]) -> set[str]:
    candidates = shortlist.get("supportcover_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Shortlist must contain a supportcover_candidates list.")
    result: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("config_id"), str):
            raise ValueError("Every shortlisted SupportCover candidate must contain config_id.")
        result.add(candidate["config_id"])
    return result


def _curate_packing_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    study: str,
    status_by_config: Mapping[str, str],
) -> list[dict[str, Any]]:
    curated: list[dict[str, Any]] = []
    for row in rows:
        if row.get("study") != study:
            continue
        config_id = str(row.get("config_id", ""))
        curated.append(
            {
                **{column: row.get(column, "") for column in PACKING_COLUMNS if column != "selection_status"},
                "selection_status": status_by_config.get(config_id, "not_retained"),
            }
        )
    return curated


def export_development_paper_results(
    *,
    packing_summary_path: str | Path,
    packing_manifest_path: str | Path,
    shortlist_path: str | Path,
    decision_path: str | Path,
    freeze_manifest_path: str | Path,
    output_root: str | Path = "paper_results",
    code_revision: str | None = None,
) -> dict[str, str]:
    """Export compact Phase-3 evidence; never copies per-example rows or predictions."""
    packing_summary = Path(packing_summary_path)
    packing_manifest = Path(packing_manifest_path)
    shortlist_file = Path(shortlist_path)
    decision_file = Path(decision_path)
    freeze_manifest_file = Path(freeze_manifest_path)
    sources = [packing_summary, packing_manifest, shortlist_file, decision_file, freeze_manifest_file]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Development publication sources are missing: " + ", ".join(missing))

    rows = _read_csv(packing_summary)
    if len(rows) != 29:
        raise ValueError(f"Expected exactly 29 Phase-3 packing-summary rows; found {len(rows)}.")
    shortlist = _read_json_object(shortlist_file)
    decision = _read_json_object(decision_file)
    freeze_manifest = _read_json_object(freeze_manifest_file)
    _require_completed_decision(decision)
    if shortlist.get("development_split_sha256") != DEVELOPMENT_SPLIT_SHA256:
        raise ValueError("Shortlist does not match the frozen development split SHA256.")

    retained_supportcover = _shortlisted_config_ids(shortlist)
    base = shortlist.get("base_supportcover")
    if not isinstance(base, dict) or not isinstance(base.get("config_id"), str):
        raise ValueError("Shortlist must contain base_supportcover.config_id.")
    supportcover_status = {config_id: "retained_for_generation" for config_id in retained_supportcover}
    supportcover_status[base["config_id"]] = "base"

    selected_mmr = float(decision["selected_mmr_lambda_relevance"])
    shortlisted_mmr = shortlist.get("mmr_lambdas")
    if not isinstance(shortlisted_mmr, list):
        raise ValueError("Shortlist must contain mmr_lambdas.")
    mmr_status: dict[str, str] = {
        f"mmr_lambda_{float(value):g}": "retained_for_generation" for value in shortlisted_mmr
    }
    mmr_status[f"mmr_lambda_{selected_mmr:g}"] = "final_selected"

    development_dir = ensure_dir(Path(output_root) / "01_development")
    artifacts = {
        "sensitivity": development_dir / "sensitivity.csv",
        "mmr_selection": development_dir / "mmr_selection.csv",
        "component_ablation": development_dir / "component_ablation.csv",
        "development_decision": development_dir / "development_decision.json",
    }
    write_csv(
        artifacts["sensitivity"],
        _curate_packing_rows(
            rows,
            study="supportcover_sensitivity",
            status_by_config=supportcover_status,
        ),
    )
    write_csv(
        artifacts["mmr_selection"],
        _curate_packing_rows(rows, study="mmr_lambda", status_by_config=mmr_status),
    )
    write_csv(
        artifacts["component_ablation"],
        _curate_packing_rows(rows, study="component_ablation", status_by_config={}),
    )
    curated_decision = dict(decision)
    curated_decision["source_shortlist"] = _relative_or_resolved(shortlist_file)
    curated_decision["source_freeze_manifest"] = _relative_or_resolved(freeze_manifest_file)
    write_json(artifacts["development_decision"], curated_decision)

    freeze_sha256 = str(freeze_manifest.get("config_sha256") or "") or None
    config_sha256 = freeze_sha256
    common_sources = [packing_summary, packing_manifest, shortlist_file, decision_file, freeze_manifest_file]
    for role, artifact in artifacts.items():
        register_paper_artifact(
            artifact_path=artifact,
            role=role,
            source_artifacts=common_sources,
            output_root=output_root,
            split_sha256=DEVELOPMENT_SPLIT_SHA256,
            config_sha256=config_sha256,
            freeze_sha256=freeze_sha256,
            code_revision=code_revision,
        )
    return {role: str(path) for role, path in artifacts.items()}


def export_protocol_paper_results(
    *,
    split_validation_path: str | Path,
    frozen_config_path: str | Path,
    freeze_manifest_path: str | Path,
    environment_path: str | Path,
    output_root: str | Path = "paper_results",
    code_revision: str | None = None,
) -> dict[str, str]:
    """Copy only real frozen protocol artifacts, with provenance, after Phase 3 freezes."""
    sources_by_role = {
        "split_manifest": Path(split_validation_path),
        "frozen_config": Path(frozen_config_path),
        "freeze_manifest": Path(freeze_manifest_path),
        "environment": Path(environment_path),
    }
    missing = [str(path) for path in sources_by_role.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen protocol sources are missing: " + ", ".join(missing))

    freeze_manifest = _read_json_object(freeze_manifest_path)
    freeze_sha256 = str(freeze_manifest.get("config_sha256") or "") or None
    protocol_dir = ensure_dir(Path(output_root) / "00_protocol")
    targets = {
        "split_manifest": protocol_dir / "split_manifest.json",
        "frozen_config": protocol_dir / "frozen_config.yaml",
        "freeze_manifest": protocol_dir / "freeze_manifest.json",
        "environment": protocol_dir / "environment.json",
    }
    for role, source in sources_by_role.items():
        shutil.copyfile(source, targets[role])
        register_paper_artifact(
            artifact_path=targets[role],
            role=role,
            source_artifacts=[source],
            output_root=output_root,
            split_sha256={
                "development": DEVELOPMENT_SPLIT_SHA256,
                "final": FINAL_SPLIT_SHA256,
            },
            config_sha256=freeze_sha256,
            freeze_sha256=freeze_sha256,
            code_revision=code_revision,
        )
    return {role: str(path) for role, path in targets.items()}
