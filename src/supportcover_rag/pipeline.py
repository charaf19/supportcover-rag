from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import psutil

from supportcover_rag.config import AppConfig
from supportcover_rag.data import load_examples
from supportcover_rag.device import elapsed_time_ms
from supportcover_rag.evaluation import aggregate_records, coverage_at_budget, exact_match_score, f1_score, support_metrics
from supportcover_rag.experiment_outputs import (
    ExperimentContext,
    ExperimentFamily,
    ExperimentOutputManager,
    merge_notes,
)
from supportcover_rag.generation import PromptInput, build_generator, build_token_counter
from supportcover_rag.io_utils import append_jsonl_rows, ensure_dir, write_csv, write_json
from supportcover_rag.logging_utils import attach_run_log
from supportcover_rag.packing import (
    SupportCoverSelector,
    apply_variant,
    build_sentence_candidates,
    pack_paragraphs,
    pack_random,
    pack_relevance_only,
)
from supportcover_rag.retrieval import BM25ParagraphRetriever
from supportcover_rag.types import HotpotExample, PackedEvidence, PredictionRecord

LOGGER = logging.getLogger(__name__)
SUPPORTED_METHODS = ("no_rag", "paragraph_topk", "relevance_only", "random_sentence", "supportcover", "supportcover_final")


@dataclass(slots=True)
class PreparedPrediction:
    example: HotpotExample
    packed: PackedEvidence
    evidence_text: str
    retrieval_latency_ms: float
    packing_latency_ms: float
    metadata: dict[str, object]


class ExperimentRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever = BM25ParagraphRetriever(
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
        )
        self.generator = build_generator(config.generation, config.prompting)
        self.execution_device = getattr(self.generator, "device", None)
        self.token_counter = build_token_counter(config.generation)
        self.process = psutil.Process()
        self.output_manager = ExperimentOutputManager(config.paths.output_root)

    def close(self) -> None:
        generator = getattr(self, "generator", None)
        if generator is None:
            return
        close_generator = getattr(generator, "close", None)
        if callable(close_generator):
            close_generator()
        self.generator = None

    def _write_suite_summary(
        self,
        *,
        family: ExperimentFamily,
        summaries: list[dict[str, float | int | str]],
    ) -> Path | None:
        if not summaries:
            return None
        first_id = str(summaries[0]["experiment_id"])
        last_id = str(summaries[-1]["experiment_id"])
        target = Path(self.config.paths.output_root) / family.value / f"{first_id}_{last_id}_comparison.csv"
        write_csv(target, summaries)
        return target

    @property
    def batch_size(self) -> int:
        return max(1, getattr(self.generator, "batch_size", self.config.generation.batch_size))

    def _build_packed_evidence(
        self,
        example: HotpotExample,
        method: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str = "full",
    ) -> tuple[PackedEvidence, float, float, dict[str, object]]:
        retrieval_start = time.perf_counter()
        retrieved = self.retriever.retrieve(example=example, top_k=retrieval_depth)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        packing_start = time.perf_counter()
        if method == "no_rag":
            packed = PackedEvidence(method=method, selected=[], token_budget=token_budget)
            candidates = []
        else:
            candidates = build_sentence_candidates(
                question=example.question,
                retrieved_paragraphs=retrieved,
                token_counter=self.token_counter,
            )
            if method == "paragraph_topk":
                packed = pack_paragraphs(
                    question=example.question,
                    retrieved_paragraphs=retrieved,
                    token_budget=token_budget,
                    token_counter=self.token_counter,
                )
            elif method == "relevance_only":
                packed = pack_relevance_only(candidates, token_budget=token_budget)
            elif method == "random_sentence":
                packed = pack_random(candidates, token_budget=token_budget, seed=self.config.seed)
            elif method == "supportcover":
                selector = apply_variant(SupportCoverSelector(self.config.supportcover), variant=variant)
                packed = selector.select(candidates, token_budget=token_budget)
            elif method == "supportcover_final":
                canonical_variant = self.config.robustness.supportcover_final_variant
                selector = apply_variant(SupportCoverSelector(self.config.supportcover), variant=canonical_variant)
                packed = selector.select(candidates, token_budget=token_budget)
            else:
                raise ValueError(f"Unsupported method: {method}")
        packing_latency_ms = (time.perf_counter() - packing_start) * 1000.0

        metadata = {
            "retrieved_titles": [paragraph.title for paragraph in retrieved],
            "num_candidates": len(candidates),
            "variant": variant,
        }
        return packed, retrieval_latency_ms, packing_latency_ms, metadata

    def _prepare_prediction(
        self,
        example: HotpotExample,
        method: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str = "full",
    ) -> PreparedPrediction:
        packed, retrieval_latency_ms, packing_latency_ms, metadata = self._build_packed_evidence(
            example=example,
            method=method,
            token_budget=token_budget,
            retrieval_depth=retrieval_depth,
            variant=variant,
        )
        evidence_text = packed.render(include_titles=self.config.prompting.include_titles)
        return PreparedPrediction(
            example=example,
            packed=packed,
            evidence_text=evidence_text,
            retrieval_latency_ms=retrieval_latency_ms,
            packing_latency_ms=packing_latency_ms,
            metadata=metadata,
        )

    def _build_prediction_record(
        self,
        prepared: PreparedPrediction,
        method: str,
        token_budget: int,
        generation_text: str,
        generated_tokens: int,
        generation_latency_ms: float,
    ) -> PredictionRecord:
        example = prepared.example
        support = support_metrics(prepared.packed.support_keys, example.supporting_facts)
        return PredictionRecord(
            example_id=example.example_id,
            method=method,
            token_budget=token_budget,
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=generation_text,
            gold_supporting_facts=example.supporting_facts,
            predicted_supporting_facts=prepared.packed.support_keys,
            answer_em=exact_match_score(generation_text, example.answer),
            answer_f1=f1_score(generation_text, example.answer),
            support_em=support["support_em"],
            support_precision=support["support_precision"],
            support_recall=support["support_recall"],
            support_f1=support["support_f1"],
            coverage_at_budget=coverage_at_budget(prepared.packed.support_keys, example.supporting_facts),
            evidence_tokens=prepared.packed.used_tokens,
            retrieval_latency_ms=prepared.retrieval_latency_ms,
            packing_latency_ms=prepared.packing_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=prepared.retrieval_latency_ms + prepared.packing_latency_ms + generation_latency_ms,
            peak_rss_mb=self.process.memory_info().rss / (1024 * 1024),
            metadata={
                **prepared.metadata,
                "evidence_text": prepared.evidence_text,
                "generated_tokens": generated_tokens,
                "question_type": example.qtype,
                "difficulty": example.level,
            },
        )

    def _generate_records_for_batch(
        self,
        prepared_batch: list[PreparedPrediction],
        method: str,
        token_budget: int,
    ) -> list[PredictionRecord]:
        batch_items = [
            PromptInput(question=prepared.example.question, evidence=prepared.evidence_text) for prepared in prepared_batch
        ]
        generation_start = time.perf_counter()
        generations = self.generator.generate_batch(batch_items)
        batch_generation_latency_ms = elapsed_time_ms(generation_start, self.execution_device)

        if len(generations) != len(prepared_batch):
            raise RuntimeError(
                f"Generator returned {len(generations)} outputs for a batch of {len(prepared_batch)} prepared examples."
            )

        generation_latency_ms = batch_generation_latency_ms / len(prepared_batch)
        return [
            self._build_prediction_record(
                prepared=prepared,
                method=method,
                token_budget=token_budget,
                generation_text=generation.text,
                generated_tokens=generation.generated_tokens,
                generation_latency_ms=generation_latency_ms,
            )
            for prepared, generation in zip(prepared_batch, generations, strict=True)
        ]

    def _format_eta(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _log_progress(self, method: str, processed: int, total: int, started_at: float) -> None:
        if processed <= 0:
            return
        elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
        avg_seconds = elapsed_seconds / processed
        eta_seconds = avg_seconds * max(total - processed, 0)
        LOGGER.info(
            "Progress | method=%s | processed=%d/%d | avg_sec_per_example=%.2f | eta=%s",
            method,
            processed,
            total,
            avg_seconds,
            self._format_eta(eta_seconds),
        )

    def _build_run_payload(
        self,
        context: ExperimentContext,
        metrics: dict[str, float | int | str] | None,
        *,
        status: str,
        notes: str,
    ) -> dict[str, float | int | str]:
        data = metrics or {}
        default_count: int | str = 0 if status == "completed" else ""
        default_metric: float | str = 0.0 if status == "completed" else ""
        return {
            "experiment_id": context.experiment_id,
            "family": context.family.value,
            "timestamp": context.timestamp,
            "status": status,
            "method": context.method,
            "model": context.model_alias,
            "dataset": context.dataset,
            "split": context.split,
            "num_examples": data.get("num_examples", default_count),
            "token_budget": context.token_budget,
            "retrieval_depth": context.retrieval_depth,
            "variant": context.variant,
            "answer_em": data.get("answer_em", default_metric),
            "answer_f1": data.get("answer_f1", default_metric),
            "support_em": data.get("support_em", default_metric),
            "support_precision": data.get("support_precision", default_metric),
            "support_recall": data.get("support_recall", default_metric),
            "support_f1": data.get("support_f1", default_metric),
            "coverage_at_budget": data.get("coverage_at_budget", default_metric),
            "evidence_tokens": data.get("evidence_tokens", default_metric),
            "retrieval_latency_ms": data.get("retrieval_latency_ms", default_metric),
            "packing_latency_ms": data.get("packing_latency_ms", default_metric),
            "generation_latency_ms": data.get("generation_latency_ms", default_metric),
            "total_latency_ms": data.get("total_latency_ms", default_metric),
            "peak_rss_mb": data.get("peak_rss_mb", default_metric),
            "output_dir": str(context.output_dir),
            "notes": notes,
        }

    def _execute_run(
        self,
        split_path: str | Path,
        output_dir: str | Path,
        method: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str = "full",
    ) -> dict[str, float | int | str]:
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(SUPPORTED_METHODS)
            raise ValueError(f"Unsupported method '{method}'. Expected one of: {supported}.")

        examples = load_examples(split_path, limit=self.config.runtime.limit)
        target_dir = ensure_dir(output_dir)
        predictions_path = target_dir / "predictions.jsonl"
        records: list[PredictionRecord] = []

        if predictions_path.exists() and not self.config.runtime.overwrite:
            raise FileExistsError(f"{predictions_path} already exists. Set runtime.overwrite=true or choose a new output directory.")
        if predictions_path.exists():
            predictions_path.unlink()

        LOGGER.info(
            "Running %s on %d examples | budget=%d | retrieval_depth=%d | variant=%s | batch_size=%d",
            method,
            len(examples),
            token_budget,
            retrieval_depth,
            variant,
            self.batch_size,
        )

        prepared_batch: list[PreparedPrediction] = []
        processed = 0
        last_logged = 0
        progress_interval = max(self.batch_size * 5, 25)
        started_at = time.perf_counter()

        for example in examples:
            prepared_batch.append(
                self._prepare_prediction(
                    example=example,
                    method=method,
                    token_budget=token_budget,
                    retrieval_depth=retrieval_depth,
                    variant=variant,
                )
            )
            if len(prepared_batch) < self.batch_size:
                continue

            batch_records = self._generate_records_for_batch(
                prepared_batch=prepared_batch,
                method=method,
                token_budget=token_budget,
            )
            records.extend(batch_records)
            append_jsonl_rows(predictions_path, [record.to_dict() for record in batch_records])
            processed += len(batch_records)
            if processed <= self.batch_size or processed == len(examples) or processed - last_logged >= progress_interval:
                self._log_progress(method=method, processed=processed, total=len(examples), started_at=started_at)
                last_logged = processed
            prepared_batch = []

        if prepared_batch:
            batch_records = self._generate_records_for_batch(
                prepared_batch=prepared_batch,
                method=method,
                token_budget=token_budget,
            )
            records.extend(batch_records)
            append_jsonl_rows(predictions_path, [record.to_dict() for record in batch_records])
            processed += len(batch_records)
            self._log_progress(method=method, processed=processed, total=len(examples), started_at=started_at)

        return aggregate_records(records)

    def run_single(
        self,
        split_path: str | Path,
        split_name: str,
        method: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str = "full",
        family: ExperimentFamily = ExperimentFamily.MAIN,
        notes: str = "",
        experiment_id: str | None = None,
    ) -> dict[str, float | int | str]:
        context = self.output_manager.prepare_run(
            config=self.config,
            family=family,
            method=method,
            split_name=split_name,
            token_budget=token_budget,
            retrieval_depth=retrieval_depth,
            variant=variant,
            notes=notes,
            experiment_id=experiment_id,
        )
        target_dir = ensure_dir(context.output_dir)
        self.output_manager.write_config_snapshot(target_dir / "config.resolved.yaml", self.config, context)

        with attach_run_log(target_dir / "run.log"):
            LOGGER.info(
                "Experiment start | id=%s | family=%s | method=%s | model=%s | split=%s | budget=%d | retrieval_depth=%d | variant=%s | output_dir=%s",
                context.experiment_id,
                context.family.value,
                context.method,
                context.model_alias,
                context.split,
                context.token_budget,
                context.retrieval_depth,
                context.variant,
                context.output_dir,
            )
            try:
                metrics = self._execute_run(
                    split_path=split_path,
                    output_dir=target_dir,
                    method=method,
                    token_budget=token_budget,
                    retrieval_depth=retrieval_depth,
                    variant=variant,
                )
                payload = self._build_run_payload(context, metrics, status="completed", notes=context.notes)
                write_json(target_dir / "metrics.json", payload)
                write_csv(target_dir / "summary.csv", [payload])
                self.output_manager.append_registry_row(payload)
                LOGGER.info(
                    "Experiment complete | id=%s | answer_f1=%.4f | support_f1=%.4f | coverage_at_budget=%.4f | total_latency_ms=%.2f",
                    context.experiment_id,
                    float(payload["answer_f1"]),
                    float(payload["support_f1"]),
                    float(payload["coverage_at_budget"]),
                    float(payload["total_latency_ms"]),
                )
                return payload
            except Exception as exc:
                failure_notes = merge_notes(context.notes, f"error: {exc}")
                payload = self._build_run_payload(context, None, status="failed", notes=failure_notes)
                write_json(target_dir / "metrics.json", payload)
                write_csv(target_dir / "summary.csv", [payload])
                self.output_manager.append_registry_row(payload)
                LOGGER.exception("Experiment failed | id=%s", context.experiment_id)
                raise

    def run_main_suite(
        self,
        split_path: str | Path,
        split_name: str,
        *,
        family: ExperimentFamily = ExperimentFamily.MAIN,
        notes: str = "",
        experiment_id: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        if experiment_id is not None and len(self.config.experiments.methods) != 1:
            raise ValueError("Use --experiment-id only when the command will produce exactly one run.")

        summaries: list[dict[str, float | int | str]] = []
        for method in self.config.experiments.methods:
            metrics = self.run_single(
                split_path=split_path,
                split_name=split_name,
                method=method,
                token_budget=self.config.supportcover.token_budget,
                retrieval_depth=self.config.retrieval.top_k_paragraphs,
                family=family,
                notes=notes,
                experiment_id=experiment_id,
            )
            summaries.append(metrics)
        summary_path = self._write_suite_summary(family=family, summaries=summaries)
        if summary_path is not None:
            LOGGER.info("Wrote comparison summary: %s", summary_path)
        return summaries

    def run_robustness_suite(
        self,
        split_path: str | Path,
        split_name: str,
        *,
        notes: str = "",
    ) -> list[dict[str, float | int | str]]:
        return self.run_robustness_study(
            self.config,
            split_path=split_path,
            split_name=split_name,
            notes=notes,
        )

    @classmethod
    def run_robustness_study(
        cls,
        config: AppConfig,
        split_path: str | Path,
        split_name: str,
        *,
        notes: str = "",
    ) -> list[dict[str, float | int | str]]:
        models = list(config.robustness.models)
        if len(models) < 2:
            raise ValueError("Configure at least two models for the robustness study.")
        if len(models) > 3:
            raise ValueError("Robustness study supports at most three models.")

        summaries: list[dict[str, float | int | str]] = []
        for model_name in models:
            model_config = replace(
                config,
                generation=replace(config.generation, model_name_or_path=model_name),
            )
            model_runner = cls(model_config)
            try:
                model_notes = merge_notes(notes, f"supportcover_final={config.robustness.supportcover_final_variant}")
                for method, variant in (("relevance_only", "full"), ("supportcover_final", "final")):
                    summaries.append(
                        model_runner.run_single(
                            split_path=split_path,
                            split_name=split_name,
                            method=method,
                            token_budget=model_config.supportcover.token_budget,
                            retrieval_depth=model_config.retrieval.top_k_paragraphs,
                            variant=variant,
                            family=ExperimentFamily.ROBUSTNESS,
                            notes=model_notes,
                        )
                    )
            finally:
                model_runner.close()

        summary_path = Path(config.paths.output_root) / ExperimentFamily.ROBUSTNESS.value / (
            f"{summaries[0]['experiment_id']}_{summaries[-1]['experiment_id']}_comparison.csv"
        )
        write_csv(summary_path, summaries)
        LOGGER.info("Wrote comparison summary: %s", summary_path)
        return summaries

    def _build_ablation_plan(
        self,
        family: ExperimentFamily | None,
    ) -> list[tuple[ExperimentFamily, str, int, int, str]]:
        if family in {ExperimentFamily.MAIN, ExperimentFamily.BASELINE, ExperimentFamily.ROBUSTNESS}:
            raise ValueError(
                "run-ablations supports only ablation_budget, ablation_depth, ablation_component, or debug families."
            )

        plans: list[tuple[ExperimentFamily, str, int, int, str]] = []
        budget_methods = [method for method in self.config.experiments.methods if method in {"relevance_only", "supportcover"}]
        if not budget_methods:
            budget_methods = ["supportcover"]
        depth_methods = [method for method in self.config.experiments.methods if method in {"relevance_only", "supportcover"}]
        if not depth_methods:
            depth_methods = ["supportcover"]

        def add_budget_runs(target_family: ExperimentFamily) -> None:
            for budget in self.config.ablations.token_budgets:
                for method in budget_methods:
                    plans.append((target_family, method, budget, self.config.retrieval.top_k_paragraphs, "full"))

        def add_depth_runs(target_family: ExperimentFamily) -> None:
            for depth in self.config.ablations.retrieval_depths:
                for method in depth_methods:
                    plans.append((target_family, method, self.config.supportcover.token_budget, depth, "full"))

        def add_component_runs(target_family: ExperimentFamily) -> None:
            for variant in self.config.ablations.variants:
                method = "supportcover" if variant != "relevance_only" else "relevance_only"
                plans.append(
                    (
                        target_family,
                        method,
                        self.config.supportcover.token_budget,
                        self.config.retrieval.top_k_paragraphs,
                        variant,
                    )
                )

        if family is None:
            add_budget_runs(ExperimentFamily.ABLATION_BUDGET)
            add_depth_runs(ExperimentFamily.ABLATION_DEPTH)
            add_component_runs(ExperimentFamily.ABLATION_COMPONENT)
            return plans
        if family is ExperimentFamily.DEBUG:
            add_budget_runs(ExperimentFamily.DEBUG)
            add_depth_runs(ExperimentFamily.DEBUG)
            add_component_runs(ExperimentFamily.DEBUG)
            return plans
        if family is ExperimentFamily.ABLATION_BUDGET:
            add_budget_runs(ExperimentFamily.ABLATION_BUDGET)
            return plans
        if family is ExperimentFamily.ABLATION_DEPTH:
            add_depth_runs(ExperimentFamily.ABLATION_DEPTH)
            return plans
        add_component_runs(ExperimentFamily.ABLATION_COMPONENT)
        return plans

    def run_ablations(
        self,
        split_path: str | Path,
        split_name: str,
        *,
        family: ExperimentFamily | None = None,
        notes: str = "",
        experiment_id: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        plans = self._build_ablation_plan(family)
        if experiment_id is not None and len(plans) != 1:
            raise ValueError("Use --experiment-id only when the command will produce exactly one run.")

        results: list[dict[str, float | int | str]] = []
        for plan_family, method, token_budget, retrieval_depth, variant in plans:
            metrics = self.run_single(
                split_path=split_path,
                split_name=split_name,
                method=method,
                token_budget=token_budget,
                retrieval_depth=retrieval_depth,
                variant=variant,
                family=plan_family,
                notes=notes,
                experiment_id=experiment_id,
            )
            results.append(metrics)
        if family is not None:
            summary_path = self._write_suite_summary(family=family, summaries=results)
            if summary_path is not None:
                LOGGER.info("Wrote comparison summary: %s", summary_path)
        return results
