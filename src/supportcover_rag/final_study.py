from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StudyStage:
    order: int
    name: str
    depends_on: tuple[str, ...]
    requires: tuple[str, ...]
    produces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StagePrerequisiteValidation:
    stage: str
    passed: bool
    missing: tuple[str, ...]


FINAL_STUDY_PLAN: tuple[StudyStage, ...] = (
    StudyStage(
        order=1,
        name="protocol_integrity",
        depends_on=(),
        requires=("baseline_manifest_schema", "explicit_development_ids", "explicit_final_ids"),
        produces=("protocol_manifest", "final_split_sha256"),
    ),
    StudyStage(
        order=2,
        name="frozen_configuration_validation",
        depends_on=("protocol_integrity",),
        requires=("final_split_sha256", "resolved_final_configuration", "frozen_config_sha256"),
        produces=("validated_frozen_configuration",),
    ),
    StudyStage(
        order=3,
        name="main_study",
        depends_on=("frozen_configuration_validation",),
        requires=("explicit_final_ids", "final_split_sha256", "frozen_config_sha256", "external_compressor_adapter"),
        produces=("main_prediction_populations", "main_aggregate_metrics"),
    ),
    StudyStage(
        order=4,
        name="statistical_comparison",
        depends_on=("main_study",),
        requires=("main_prediction_populations", "matched_final_prediction_populations"),
        produces=("paired_statistical_comparisons",),
    ),
    StudyStage(
        order=5,
        name="token_budget_robustness",
        depends_on=("frozen_configuration_validation", "main_study"),
        requires=("main_study_completion", "frozen_budget_protocol", "explicit_final_ids", "final_split_sha256"),
        produces=("budget_robustness_predictions", "budget_robustness_metrics"),
    ),
    StudyStage(
        order=6,
        name="model_robustness",
        depends_on=("frozen_configuration_validation", "main_study"),
        requires=("main_study_completion", "frozen_model_protocol", "explicit_final_ids", "final_split_sha256"),
        produces=("model_robustness_predictions", "model_robustness_metrics"),
    ),
    StudyStage(
        order=7,
        name="cross_dataset_robustness",
        depends_on=("frozen_configuration_validation", "main_study"),
        requires=("main_study_completion", "multihop_qa_adapter", "cross_dataset_final_ids", "frozen_cross_dataset_protocol"),
        produces=("cross_dataset_predictions", "cross_dataset_metrics"),
    ),
    StudyStage(
        order=8,
        name="global_retrieval_evaluation",
        depends_on=("protocol_integrity", "frozen_configuration_validation"),
        requires=(
            "frozen_config_sha256",
            "resolved_global_retrieval_protocol",
            "deterministic_global_corpus",
            "global_corpus_sha256",
            "global_index_corpus_binding",
            "global_retrieval_query_role",
            "canonical_gold_support",
        ),
        produces=("global_retrieval_diagnostics", "global_retrieval_metrics"),
    ),
    StudyStage(
        order=9,
        name="systems_benchmark",
        depends_on=("frozen_configuration_validation",),
        requires=("validated_frozen_configuration", "environment_manifest", "benchmark_descriptor"),
        produces=("raw_latency_samples", "systems_benchmark_summary"),
    ),
    StudyStage(
        order=10,
        name="mechanism_coverage_diagnostics",
        depends_on=("main_study",),
        requires=("main_prediction_populations", "packed_evidence_metadata", "canonical_gold_support"),
        produces=("coverage_diagnostics", "coverage_correlations", "blinded_error_analysis_records"),
    ),
    StudyStage(
        order=11,
        name="reproducibility_verification",
        depends_on=(
            "statistical_comparison",
            "token_budget_robustness",
            "model_robustness",
            "cross_dataset_robustness",
            "global_retrieval_evaluation",
            "systems_benchmark",
            "mechanism_coverage_diagnostics",
        ),
        requires=(
            "protocol_manifest",
            "validated_frozen_configuration",
            "paired_statistical_comparisons",
            "budget_robustness_metrics",
            "model_robustness_metrics",
            "cross_dataset_metrics",
            "global_retrieval_metrics",
            "systems_benchmark_summary",
            "coverage_diagnostics",
            "blinded_error_analysis_records",
        ),
        produces=("reproducibility_report",),
    ),
)


def validate_plan_structure(plan: Sequence[StudyStage] = FINAL_STUDY_PLAN) -> None:
    names: set[str] = set()
    expected_order = 1
    for stage in plan:
        if stage.order != expected_order:
            raise ValueError(f"Final-study stage '{stage.name}' has order {stage.order}; expected {expected_order}.")
        if stage.name in names:
            raise ValueError(f"Duplicate final-study stage name: {stage.name}")
        missing_dependencies = [dependency for dependency in stage.depends_on if dependency not in names]
        if missing_dependencies:
            raise ValueError(
                f"Stage '{stage.name}' depends on later or unknown stages: {', '.join(missing_dependencies)}"
            )
        names.add(stage.name)
        expected_order += 1


def validate_stage_prerequisites(
    stage: StudyStage,
    available_artifacts: Iterable[str],
) -> StagePrerequisiteValidation:
    available = set(available_artifacts)
    missing = tuple(requirement for requirement in stage.requires if requirement not in available)
    return StagePrerequisiteValidation(stage=stage.name, passed=not missing, missing=missing)


def expected_artifact_descriptors(
    plan: Sequence[StudyStage] = FINAL_STUDY_PLAN,
) -> tuple[tuple[str, str], ...]:
    validate_plan_structure(plan)
    return tuple((artifact, stage.name) for stage in plan for artifact in stage.produces)
