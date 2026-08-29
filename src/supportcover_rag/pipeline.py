from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psutil

from supportcover_rag.config import AppConfig
from supportcover_rag.data import load_examples, load_examples_by_ids
from supportcover_rag.device import elapsed_time_ms
from supportcover_rag.evaluation import aggregate_records, coverage_at_budget, exact_match_score, f1_score, support_metrics
from supportcover_rag.experiment_outputs import (
    ExperimentContext,
    ExperimentFamily,
    ExperimentOutputManager,
    merge_notes,
)
from supportcover_rag.external_baselines import EvidenceCompressor
from supportcover_rag.generation import PromptInput, build_generator, build_token_counter
from supportcover_rag.io_utils import append_jsonl_rows, ensure_dir, read_jsonl, write_csv, write_json
from supportcover_rag.logging_utils import attach_run_log
from supportcover_rag.packing import (
    SupportCoverSelector,
    apply_variant,
    build_sentence_candidates,
    pack_greedy_query_cover,
    pack_mmr,
    pack_paragraphs,
    pack_random,
    pack_relevance_only,
)
from supportcover_rag.retrieval import BM25ParagraphRetriever
from supportcover_rag.splits import load_json_ids, ordered_ids_sha256, validate_unique_ids
from supportcover_rag.types import HotpotExample, PackedEvidence, PredictionRecord

LOGGER = logging.getLogger(__name__)
SUPPORTED_METHODS = (
    "no_rag",
    "paragraph_topk",
    "relevance_only",
    "random_sentence",
    "mmr_sentence",
    "greedy_query_cover",
    "external_compressor",
    "supportcover",
    "supportcover_final",
)


@dataclass(slots=True)
class PreparedPrediction:
    example: HotpotExample
    packed: PackedEvidence
    evidence_text: str
    retrieval_latency_ms: float
    packing_latency_ms: float
    metadata: dict[str, object]


class ExperimentRunner:
    def __init__(self, config: AppConfig, *, external_compressor: EvidenceCompressor | None = None) -> None:
        self.config = config
        self.external_compressor = external_compressor
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
        if method == "external_compressor":
            if self.external_compressor is None:
                raise RuntimeError("The external_compressor method requires an injected EvidenceCompressor adapter.")
            candidates = []
            packed = self.external_compressor.compress(
                question=example.question,
                retrieved_paragraphs=retrieved,
                token_budget=token_budget,
            )
            if not isinstance(packed, PackedEvidence):
                raise TypeError("EvidenceCompressor.compress() must return PackedEvidence.")
            if packed.used_tokens > token_budget:
                raise ValueError(
                    f"External compressor exceeded the token budget: {packed.used_tokens} > {token_budget}."
                )
        elif method == "no_rag":
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
            elif method == "mmr_sentence":
                packed = pack_mmr(
                    candidates,
                    token_budget=token_budget,
                    lambda_relevance=self.config.retrieval.mmr_lambda_relevance,
                )
            elif method == "greedy_query_cover":
                packed = pack_greedy_query_cover(candidates, token_budget=token_budget)
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
        payload: dict[str, float | int | str] = {
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
        optional_provenance = {
            "config_sha256": context.config_sha256,
            "code_revision": context.code_revision,
            "split_sha256": context.split_sha256,
        }
        payload.update(
            {key: value for key, value in optional_provenance.items() if value is not None}
        )
        return payload

    def _configured_explicit_ids(self) -> tuple[list[str] | None, str | None]:
        role = self.config.split.role.strip().lower()
        if role == "final" and self.config.runtime.limit is not None:
            raise ValueError("runtime.limit must be null for a final split.")

        ids_file = self.config.split.ids_file.strip()
        if not ids_file:
            if role == "final":
                raise ValueError("A split.ids_file is required when split.role is 'final'.")
            return None, None

        ids = load_json_ids(ids_file)
        return ids, ordered_ids_sha256(ids)

    def _resolved_frozen_config_sha256(self, supplied_sha256: str | None) -> str | None:
        configured_sha256 = self.config.freeze.sha256
        supplied = supplied_sha256.strip().lower() if supplied_sha256 else None
        configured = configured_sha256.strip().lower() if configured_sha256 else None
        if supplied and configured and supplied != configured:
            raise ValueError("Supplied config SHA256 does not match freeze.sha256.")

        resolved = supplied or configured
        if resolved is not None and (len(resolved) != 64 or any(char not in "0123456789abcdef" for char in resolved)):
            raise ValueError("Frozen config SHA256 must contain exactly 64 hexadecimal characters.")

        role = self.config.split.role.strip().lower()
        if role in {"final", "test"} and self.config.freeze.require_sha256 and resolved is None:
            raise ValueError(
                f"Split role '{role}' requires an explicitly supplied frozen config SHA256."
            )
        return resolved

    def _validate_tuning_role(self, operation: str) -> None:
        role = self.config.split.role.strip().lower()
        if role in {"final", "test"}:
            raise ValueError(f"{operation} cannot use split role '{role}'.")

    @staticmethod
    def _prediction_record_from_row(row: dict[str, Any], row_number: int) -> PredictionRecord:
        try:
            return PredictionRecord(**row)
        except TypeError as exc:
            raise ValueError(f"Malformed prediction record at row {row_number}: {exc}") from exc

    def _execute_run(
        self,
        split_path: str | Path,
        output_dir: str | Path,
        method: str,
        token_budget: int,
        retrieval_depth: int,
        variant: str = "full",
        explicit_ids: list[str] | None = None,
    ) -> dict[str, float | int | str]:
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(SUPPORTED_METHODS)
            raise ValueError(f"Unsupported method '{method}'. Expected one of: {supported}.")

        if explicit_ids is None:
            explicit_ids, _ = self._configured_explicit_ids()
        elif self.config.split.role.strip().lower() == "final" and self.config.runtime.limit is not None:
            raise ValueError("runtime.limit must be null for a final split.")

        if explicit_ids is None:
            examples = load_examples(split_path, limit=self.config.runtime.limit)
        else:
            examples = load_examples_by_ids(split_path, explicit_ids)

        target_dir = ensure_dir(output_dir)
        predictions_path = target_dir / "predictions.jsonl"
        records: list[PredictionRecord] = []
        records_by_id: dict[str, PredictionRecord] = {}
        expected_ids = [example.example_id for example in examples]
        completed_ids: set[str] = set()

        if self.config.runtime.resume:
            validate_unique_ids(expected_ids)
            if predictions_path.exists():
                expected_id_set = set(expected_ids)
                for row_number, row in enumerate(read_jsonl(predictions_path), start=1):
                    example_id = row.get("example_id")
                    if not isinstance(example_id, str):
                        raise ValueError(f"Prediction row {row_number} is missing a string example_id.")
                    if example_id in records_by_id:
                        raise ValueError(f"Duplicate example ID in existing predictions: {example_id}")
                    if example_id not in expected_id_set:
                        raise ValueError(f"Unexpected example ID in existing predictions: {example_id}")
                    record = self._prediction_record_from_row(row, row_number)
                    if record.method != method:
                        raise ValueError(
                            f"Existing prediction for {example_id} uses method '{record.method}', expected '{method}'."
                        )
                    if record.token_budget != token_budget:
                        raise ValueError(
                            f"Existing prediction for {example_id} uses token budget {record.token_budget}, "
                            f"expected {token_budget}."
                        )
                    records.append(record)
                    records_by_id[example_id] = record
                completed_ids = set(records_by_id)
        else:
            if predictions_path.exists() and not self.config.runtime.overwrite:
                raise FileExistsError(
                    f"{predictions_path} already exists. Set runtime.overwrite=true or choose a new output directory."
                )
            if predictions_path.exists():
                predictions_path.unlink()

        examples_to_run = [example for example in examples if example.example_id not in completed_ids]
        if completed_ids:
            LOGGER.info(
                "Resuming %s with %d/%d examples already completed.",
                method,
                len(completed_ids),
                len(examples),
            )

        LOGGER.info(
            "Running %s on %d examples | budget=%d | retrieval_depth=%d | variant=%s | batch_size=%d",
            method,
            len(examples_to_run),
            token_budget,
            retrieval_depth,
            variant,
            self.batch_size,
        )

        prepared_batch: list[PreparedPrediction] = []
        processed = len(completed_ids)
        last_logged = processed
        progress_interval = max(self.batch_size * 5, 25)
        started_at = time.perf_counter()

        for example in examples_to_run:
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
            records_by_id.update((record.example_id, record) for record in batch_records)
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
            records_by_id.update((record.example_id, record) for record in batch_records)
            append_jsonl_rows(predictions_path, [record.to_dict() for record in batch_records])
            processed += len(batch_records)
            self._log_progress(method=method, processed=processed, total=len(examples), started_at=started_at)

        if self.config.runtime.resume:
            actual_ids = set(records_by_id)
            expected_id_set = set(expected_ids)
            missing_ids = [item_id for item_id in expected_ids if item_id not in actual_ids]
            unexpected_ids = sorted(actual_ids - expected_id_set)
            if missing_ids or unexpected_ids or len(records_by_id) != len(expected_ids):
                problems: list[str] = []
                if missing_ids:
                    problems.append(f"missing IDs: {', '.join(missing_ids)}")
                if unexpected_ids:
                    problems.append(f"unexpected IDs: {', '.join(unexpected_ids)}")
                if len(records_by_id) != len(expected_ids):
                    problems.append(
                        f"record count {len(records_by_id)} does not match expected count {len(expected_ids)}"
                    )
                raise RuntimeError("Resume produced an invalid logical prediction set: " + "; ".join(problems))
            records = [records_by_id[item_id] for item_id in expected_ids]

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
        config_sha256: str | None = None,
        code_revision: str | None = None,
    ) -> dict[str, float | int | str]:
        explicit_ids, split_sha256 = self._configured_explicit_ids()
        resolved_config_sha256 = self._resolved_frozen_config_sha256(config_sha256)
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
            config_sha256=resolved_config_sha256,
            code_revision=code_revision,
            split_sha256=split_sha256,
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
                    explicit_ids=explicit_ids,
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
        config_sha256: str | None = None,
        code_revision: str | None = None,
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
                config_sha256=config_sha256,
                code_revision=code_revision,
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
        config_sha256: str | None = None,
        code_revision: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        return self.run_robustness_study(
            self.config,
            split_path=split_path,
            split_name=split_name,
            notes=notes,
            config_sha256=config_sha256,
            code_revision=code_revision,
        )

    @classmethod
    def run_robustness_study(
        cls,
        config: AppConfig,
        split_path: str | Path,
        split_name: str,
        *,
        notes: str = "",
        config_sha256: str | None = None,
        code_revision: str | None = None,
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
                            config_sha256=config_sha256,
                            code_revision=code_revision,
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
        budget_methods = [
            method
            for method in self.config.experiments.methods
            if method in {"relevance_only", "mmr_sentence", "supportcover", "supportcover_final"}
        ]
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
        config_sha256: str | None = None,
        code_revision: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        if family is not ExperimentFamily.ABLATION_BUDGET:
            self._validate_tuning_role("Tuning/ablation execution")
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
                config_sha256=config_sha256,
                code_revision=code_revision,
            )
            results.append(metrics)
        if family is not None:
            summary_path = self._write_suite_summary(family=family, summaries=results)
            if summary_path is not None:
                LOGGER.info("Wrote comparison summary: %s", summary_path)
        return results
