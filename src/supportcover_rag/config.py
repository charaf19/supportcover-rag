from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    data_root: str = "./data"
    output_root: str = "./outputs"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True)
class RuntimeConfig:
    limit: int | None = None
    overwrite: bool = False
    resume: bool = False


@dataclass(slots=True)
class FreezeConfig:
    manifest_file: str = ""
    sha256: str | None = None
    require_sha256: bool = False


@dataclass(slots=True)
class RawDataConfig:
    dataset_path: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])


@dataclass(slots=True)
class SplitConfig:
    ids_file: str = ""
    role: str = ""
    stratify_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalConfig:
    method: str = "bm25"
    top_k_paragraphs: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    mmr_lambda_relevance: float = 0.5


@dataclass(slots=True)
class SupportCoverConfig:
    token_budget: int = 160
    stop_threshold: float = 0.0
    alpha_relevance: float = 1.0
    beta_coverage: float = 1.2
    gamma_redundancy: float = 0.6
    delta_token_cost: float = 0.15
    title_bonus: float = 0.3


@dataclass(slots=True)
class PromptingConfig:
    include_titles: bool = True
    allow_abstain: bool = True
    system_instruction: str = (
        "Answer using only the provided evidence. "
        "If the evidence is insufficient, output exactly: insufficient evidence."
    )
    user_instruction: str = "Return only the short final answer. Do not explain."


@dataclass(slots=True)
class GenerationConfig:
    backend: str = "transformers"
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    think: bool = False
    stream: bool = False
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 4
    temperature: float = 0.0
    max_new_tokens: int = 12
    do_sample: bool = False
    trust_remote_code: bool = False


@dataclass(slots=True)
class ExperimentsConfig:
    split: str = "validation"
    methods: list[str] = field(
        default_factory=lambda: ["no_rag", "paragraph_topk", "relevance_only", "random_sentence", "supportcover"]
    )


@dataclass(slots=True)
class AblationsConfig:
    token_budgets: list[int] = field(default_factory=lambda: [64, 96, 128, 160, 192])
    retrieval_depths: list[int] = field(default_factory=lambda: [3, 5, 10])
    variants: list[str] = field(
        default_factory=lambda: [
            "full",
            "relevance_only",
            "no_query_coverage",
            "no_title_gain",
            "no_redundancy",
            "no_token_penalty",
        ]
    )


@dataclass(slots=True)
class SensitivityConfig:
    beta: list[float] = field(default_factory=lambda: [0.3, 0.6, 1.2, 1.8, 2.4])
    title: list[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.45, 0.60])
    delta: list[float] = field(default_factory=lambda: [0.0, 0.075, 0.15, 0.225, 0.30])
    gamma: list[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.60, 0.90])


@dataclass(slots=True)
class RobustnessConfig:
    models: list[str] = field(default_factory=list)
    supportcover_final_variant: str = "no_redundancy"


@dataclass(slots=True)
class ErrorAnalysisConfig:
    frozen_setup_config: str = "configs/phase6_model_robustness.yaml"
    output_dir: str = "./outputs/error_analysis"
    canonical_method: str = "supportcover_final"
    comparator_method: str = "relevance_only"
    source_runs: dict[str, str] = field(default_factory=dict)
    canonical_sample_size: int = 30
    comparison_sample_size: int = 10
    representative_examples: int = 5
    taxonomy: list[str] = field(
        default_factory=lambda: [
            "support_missing",
            "support_present_answer_wrong",
            "formatting_mismatch",
            "hallucination",
            "multi_hop_reasoning_failure",
            "insufficient_evidence_forced_answer",
            "other",
        ]
    )


@dataclass(slots=True)
class SystemsSummaryConfig:
    frozen_setup_config: str = "configs/phase6_model_robustness.yaml"
    output_dir: str = "./outputs/systems"
    comparator_method: str = "relevance_only"
    canonical_method: str = "supportcover_final"
    source_runs: dict[str, str] = field(default_factory=dict)
    figure_artifact_name: str = "phase8_latency_breakdown.csv"


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: [
        "answer_em",
        "answer_f1",
        "support_em",
        "support_f1",
        "coverage_at_budget",
        "evidence_tokens",
        "total_latency_ms",
    ])


@dataclass(slots=True)
class AppConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    raw_data: RawDataConfig = field(default_factory=RawDataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    supportcover: SupportCoverConfig = field(default_factory=SupportCoverConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)
    ablations: AblationsConfig = field(default_factory=AblationsConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    error_analysis: ErrorAnalysisConfig = field(default_factory=ErrorAnalysisConfig)
    systems_summary: SystemsSummaryConfig = field(default_factory=SystemsSummaryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


_DEF_TYPE_MAP = {
    "paths": PathsConfig,
    "logging": LoggingConfig,
    "runtime": RuntimeConfig,
    "freeze": FreezeConfig,
    "raw_data": RawDataConfig,
    "split": SplitConfig,
    "retrieval": RetrievalConfig,
    "supportcover": SupportCoverConfig,
    "prompting": PromptingConfig,
    "generation": GenerationConfig,
    "experiments": ExperimentsConfig,
    "ablations": AblationsConfig,
    "sensitivity": SensitivityConfig,
    "robustness": RobustnessConfig,
    "error_analysis": ErrorAnalysisConfig,
    "systems_summary": SystemsSummaryConfig,
    "evaluation": EvaluationConfig,
}


def _build_nested(config_dict: dict[str, Any]) -> AppConfig:
    kwargs: dict[str, Any] = {}
    for key, value in config_dict.items():
        if key in _DEF_TYPE_MAP and isinstance(value, dict):
            nested_value = dict(value)
            if key == "split":
                stratify_by = nested_value.get("stratify_by")
                if stratify_by is None:
                    nested_value["stratify_by"] = []
                elif isinstance(stratify_by, str):
                    nested_value["stratify_by"] = [stratify_by]
            kwargs[key] = _DEF_TYPE_MAP[key](**nested_value)
        else:
            kwargs[key] = value
    return AppConfig(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _build_nested(payload)
