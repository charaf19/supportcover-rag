from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from supportcover_rag.config import AppConfig
from supportcover_rag.io_utils import append_csv_row, ensure_dir, read_csv_rows, write_csv, write_json, write_yaml


class ExperimentFamily(str, Enum):
    MAIN = "main"
    BASELINE = "baseline"
    ABLATION_BUDGET = "ablation_budget"
    ABLATION_DEPTH = "ablation_depth"
    ABLATION_COMPONENT = "ablation_component"
    ROBUSTNESS = "robustness"
    DEBUG = "debug"


VALID_EXPERIMENT_FAMILIES = tuple(family.value for family in ExperimentFamily)
REGISTRY_FIELDNAMES = [
    "experiment_id",
    "family",
    "timestamp",
    "status",
    "method",
    "model",
    "dataset",
    "split",
    "num_examples",
    "token_budget",
    "retrieval_depth",
    "variant",
    "answer_em",
    "answer_f1",
    "support_f1",
    "coverage_at_budget",
    "total_latency_ms",
    "output_dir",
    "notes",
]

_EXPERIMENT_ID_PATTERN = re.compile(r"^(EXP|DBG)(\d{3,})$")
_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")
_MODEL_ALIAS_OVERRIDES = {
    "Qwen/Qwen3-4B-Instruct-2507": "qwen",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama",
    "microsoft/Phi-3-mini-4k-instruct": "phi",
    "microsoft/Phi-3.5-mini-instruct": "phi",
    "google/gemma-2-2b-it": "gemma",
    "google/gemma-2-9b-it": "gemma",
}
_MODEL_ALIAS_KEYWORDS = (
    ("tinyllama", "tinyllama"),
    ("qwen", "qwen"),
    ("phi", "phi"),
    ("gemma", "gemma"),
)
_SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "val": "val",
    "train": "train",
    "test": "test",
}


@dataclass(slots=True)
class ExperimentContext:
    experiment_id: str
    family: ExperimentFamily
    method: str
    model_alias: str
    dataset: str
    split: str
    token_budget: int
    retrieval_depth: int
    variant: str
    notes: str
    timestamp: str
    output_dir: Path
    config_sha256: str | None = None
    code_revision: str | None = None
    split_sha256: str | None = None

    def run_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "family": self.family.value,
            "method": self.method,
            "model_alias": self.model_alias,
            "dataset": self.dataset,
            "split": self.split,
            "token_budget": self.token_budget,
            "retrieval_depth": self.retrieval_depth,
            "variant": self.variant,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
        }
        optional_metadata = {
            "config_sha256": self.config_sha256,
            "code_revision": self.code_revision,
            "split_sha256": self.split_sha256,
        }
        metadata.update({key: value for key, value in optional_metadata.items() if value is not None})
        return metadata


def parse_experiment_family(
    value: str | ExperimentFamily | None,
    *,
    default: ExperimentFamily | None = None,
) -> ExperimentFamily | None:
    if value is None:
        return default
    if isinstance(value, ExperimentFamily):
        return value
    normalized = value.strip().lower()
    try:
        return ExperimentFamily(normalized)
    except ValueError as exc:
        supported = ", ".join(VALID_EXPERIMENT_FAMILIES)
        raise ValueError(f"Unsupported experiment family '{value}'. Expected one of: {supported}.") from exc


def validate_experiment_id(experiment_id: str, family: ExperimentFamily) -> str:
    normalized = experiment_id.strip().upper()
    match = _EXPERIMENT_ID_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("Experiment ids must look like EXP001 or DBG001.")

    expected_prefix = "DBG" if family is ExperimentFamily.DEBUG else "EXP"
    if match.group(1) != expected_prefix:
        if family is ExperimentFamily.DEBUG:
            raise ValueError("Debug runs must use DBG ids.")
        raise ValueError("Paper-grade runs must use EXP ids.")
    return normalized


def resolve_model_alias(model_name_or_path: str) -> str:
    normalized = model_name_or_path.strip()
    if normalized in _MODEL_ALIAS_OVERRIDES:
        return _MODEL_ALIAS_OVERRIDES[normalized]

    lowered = normalized.lower()
    for needle, alias in _MODEL_ALIAS_KEYWORDS:
        if needle in lowered:
            return alias

    tail = normalized.rsplit("/", maxsplit=1)[-1]
    alias = _sanitize_name(tail)
    return alias or "model"


def resolve_split_alias(split_name: str) -> str:
    return _SPLIT_ALIASES.get(split_name.strip().lower(), _sanitize_name(split_name))


def build_dataset_alias(config: AppConfig) -> str:
    dataset_name = config.raw_data.dataset_path.rsplit("/", maxsplit=1)[-1]
    return _sanitize_name(f"{dataset_name}_{config.raw_data.dataset_config}")


def build_run_folder_name(
    *,
    experiment_id: str,
    method: str,
    model_alias: str,
    split: str,
    token_budget: int,
    retrieval_depth: int,
    variant: str,
) -> str:
    return (
        f"{experiment_id}_{_sanitize_name(method)}_{_sanitize_name(model_alias)}_{_sanitize_name(split)}"
        f"_b{int(token_budget)}_d{int(retrieval_depth)}_{_sanitize_name(variant)}"
    )


def merge_notes(notes: str, extra: str) -> str:
    base = notes.strip()
    suffix = extra.strip()
    if not base:
        return suffix
    if not suffix:
        return base
    return f"{base} | {suffix}"


class ExperimentOutputManager:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.registry_dir = self.output_root / "registry"
        self.registry_path = self.registry_dir / "experiments.csv"
        self.latest_path = self.registry_dir / "latest.json"

    def ensure_layout(self) -> None:
        ensure_dir(self.registry_dir)
        for family in ExperimentFamily:
            ensure_dir(self.output_root / family.value)

    def prepare_run(
        self,
        *,
        config: AppConfig,
        family: ExperimentFamily,
        method: str,
        split_name: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str,
        notes: str = "",
        experiment_id: str | None = None,
        config_sha256: str | None = None,
        code_revision: str | None = None,
        split_sha256: str | None = None,
    ) -> ExperimentContext:
        self.ensure_layout()

        resolved_id = validate_experiment_id(experiment_id, family) if experiment_id is not None else self.next_id(family)

        model_alias = resolve_model_alias(config.generation.model_name_or_path)
        split_alias = resolve_split_alias(split_name)
        folder_name = build_run_folder_name(
            experiment_id=resolved_id,
            method=method,
            model_alias=model_alias,
            split=split_alias,
            token_budget=token_budget,
            retrieval_depth=retrieval_depth,
            variant=variant,
        )
        output_dir = self.output_root / family.value / folder_name
        if output_dir.exists():
            if not config.runtime.resume:
                raise FileExistsError(f"{output_dir} already exists.")
            snapshot_path = output_dir / "config.resolved.yaml"
            if not snapshot_path.is_file():
                raise FileNotFoundError(f"Cannot resume without the resolved config snapshot: {snapshot_path}")
            with snapshot_path.open("r", encoding="utf-8") as handle:
                snapshot = yaml.safe_load(handle) or {}
            existing_run = snapshot.pop("run", None)
            if snapshot != asdict(config):
                raise ValueError(f"Cannot resume {resolved_id}: resolved configuration has changed.")
            if not isinstance(existing_run, dict) or existing_run.get("experiment_id") != resolved_id:
                raise ValueError(f"Cannot resume {resolved_id}: run metadata is missing or inconsistent.")
            timestamp = str(existing_run.get("timestamp") or datetime.now(timezone.utc).isoformat())
            notes = str(existing_run.get("notes") or notes).strip()
            config_sha256 = existing_run.get("config_sha256") or config_sha256
            code_revision = existing_run.get("code_revision") or code_revision
            split_sha256 = existing_run.get("split_sha256") or split_sha256
        else:
            if self.experiment_id_exists(resolved_id):
                raise ValueError(f"Experiment id '{resolved_id}' is already in use.")
            timestamp = datetime.now(timezone.utc).isoformat()

        return ExperimentContext(
            experiment_id=resolved_id,
            family=family,
            method=method,
            model_alias=model_alias,
            dataset=build_dataset_alias(config),
            split=split_alias,
            token_budget=token_budget,
            retrieval_depth=retrieval_depth,
            variant=variant,
            notes=notes.strip(),
            timestamp=timestamp,
            output_dir=output_dir,
            config_sha256=config_sha256,
            code_revision=code_revision,
            split_sha256=split_sha256,
        )

    def next_id(self, family: ExperimentFamily) -> str:
        prefix = "DBG" if family is ExperimentFamily.DEBUG else "EXP"
        next_value = max(self._existing_id_numbers(prefix), default=0) + 1
        return f"{prefix}{next_value:03d}"

    def experiment_id_exists(self, experiment_id: str) -> bool:
        if any(row.get("experiment_id") == experiment_id for row in read_csv_rows(self.registry_path)):
            return True
        for family in ExperimentFamily:
            family_dir = self.output_root / family.value
            if not family_dir.exists():
                continue
            if any(path.is_dir() for path in family_dir.glob(f"{experiment_id}_*")):
                return True
        return False

    def write_config_snapshot(self, path: str | Path, config: AppConfig, context: ExperimentContext) -> None:
        payload = asdict(config)
        payload["run"] = context.run_metadata()
        write_yaml(path, payload)

    def append_registry_row(self, row: dict[str, Any]) -> None:
        rows = read_csv_rows(self.registry_path)
        matching_indexes = [
            index for index, existing in enumerate(rows) if existing.get("experiment_id") == row.get("experiment_id")
        ]
        if len(matching_indexes) > 1:
            raise ValueError(f"Registry contains duplicate experiment id: {row.get('experiment_id')}")
        if matching_indexes:
            rows[matching_indexes[0]] = {field: row.get(field, "") for field in REGISTRY_FIELDNAMES}
            write_csv(self.registry_path, rows)
        else:
            append_csv_row(self.registry_path, row, REGISTRY_FIELDNAMES)
        write_json(self.latest_path, row)

    def _existing_id_numbers(self, prefix: str) -> list[int]:
        numbers: list[int] = []
        for row in read_csv_rows(self.registry_path):
            experiment_id = row.get("experiment_id", "")
            match = _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id)
            if match is not None and match.group(1) == prefix:
                numbers.append(int(match.group(2)))

        for family in ExperimentFamily:
            family_dir = self.output_root / family.value
            if not family_dir.exists():
                continue
            for path in family_dir.iterdir():
                if not path.is_dir():
                    continue
                experiment_id = path.name.split("_", maxsplit=1)[0]
                match = _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id)
                if match is not None and match.group(1) == prefix:
                    numbers.append(int(match.group(2)))
        return numbers


def _sanitize_name(value: str) -> str:
    normalized = _SANITIZE_PATTERN.sub("_", value.strip().lower()).strip("_")
    return normalized or "unknown"
