from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from supportcover_rag.freeze import canonical_json, canonical_sha256
from supportcover_rag.retrieval import (
    CONTROLLED_CONTEXT,
    GLOBAL_CORPUS,
    RETRIEVAL_MODES,
    GlobalBM25Index,
    GlobalCorpusRetriever,
)
from supportcover_rag.robustness import validate_freeze_boundary
from supportcover_rag.types import RetrievedParagraph, SupportKey


GLOBAL_RETRIEVAL_ROLE = "global_retrieval_evaluation"
CORPUS_PROTOCOLS = (
    "benchmark_defined_corpus",
    "frozen_union_of_benchmark_contexts",
    "externally_defined_corpus",
)
CORPUS_MANIFEST_SCHEMA_VERSION = 1
RETRIEVAL_MANIFEST_SCHEMA_VERSION = 1
UNRESOLVED = {"", "UNRESOLVED", "__PREREGISTER_LATER__", "__FROM_PHASE3_FREEZE__"}
RETRIEVAL_METRIC_NAMES = (
    "support_paragraph_recall",
    "all_support_paragraphs_retrieved_rate",
    "at_least_one_support_retrieved_rate",
    "mrr",
    "both_gold_supporting_paragraphs_retrieved_rate",
    "retrieved_candidate_count",
)
FORBIDDEN_CORPUS_SELECTION_INPUTS = {
    "gold_answers",
    "support_annotations",
    "answer_f1",
    "retrieval_outcomes",
}


@dataclass(frozen=True, slots=True)
class RetrievalReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievalReadinessReport:
    checks: tuple[RetrievalReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_scope": "metadata_only_no_final_examples",
            "checks": [asdict(check) for check in self.checks],
            "global_retrieval_execution": "READY" if self.ready else "BLOCKED",
        }


@dataclass(frozen=True, slots=True)
class FrozenRetrievedCandidates:
    example_id: str
    query_role: str
    query_population_sha256: str
    corpus_id: str
    corpus_sha256: str
    retrieval_mode: str
    retrieval_method: str
    retrieval_parameters: Mapping[str, Any]
    tokenizer_identity: str
    top_k: int
    paragraphs: tuple[RetrievedParagraph, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "paragraphs": [asdict(paragraph) for paragraph in self.paragraphs],
        }


def write_retrieval_cache(
    path: str | Path, records: Sequence[FrozenRetrievedCandidates]
) -> None:
    if not records:
        raise ValueError("Retrieval cache requires at least one query record.")
    example_ids = [record.example_id for record in records]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Retrieval cache contains duplicate example IDs.")
    protocol_fields = (
        "query_role",
        "query_population_sha256",
        "corpus_id",
        "corpus_sha256",
        "retrieval_mode",
        "retrieval_method",
        "retrieval_parameters",
        "tokenizer_identity",
        "top_k",
    )
    reference = records[0]
    for record in records[1:]:
        changed = [
            field for field in protocol_fields
            if canonical_json(getattr(record, field)) != canonical_json(getattr(reference, field))
        ]
        if changed:
            raise ValueError("Retrieval cache mixes protocols: " + ", ".join(changed))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: item.example_id):
            handle.write(canonical_json(record.to_dict()) + "\n")


def load_retrieval_cache(path: str | Path) -> list[FrozenRetrievedCandidates]:
    records: list[FrozenRetrievedCandidates] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Retrieval cache line {line_number} must be an object.")
            paragraphs = tuple(
                RetrievedParagraph(
                    title=str(item["title"]),
                    sentences=list(item["sentences"]),
                    text=str(item["text"]),
                    rank=int(item["rank"]),
                    score=float(item["score"]),
                    corpus_id=item.get("corpus_id"),
                    document_id=item.get("document_id"),
                    paragraph_id=item.get("paragraph_id"),
                    source_dataset=item.get("source_dataset"),
                    source_split=item.get("source_split"),
                    source_record_id=item.get("source_record_id"),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in value["paragraphs"]
            )
            records.append(
                FrozenRetrievedCandidates(
                    example_id=str(value["example_id"]),
                    query_role=str(value["query_role"]),
                    query_population_sha256=str(value["query_population_sha256"]),
                    corpus_id=str(value["corpus_id"]),
                    corpus_sha256=str(value["corpus_sha256"]),
                    retrieval_mode=str(value["retrieval_mode"]),
                    retrieval_method=str(value["retrieval_method"]),
                    retrieval_parameters=dict(value["retrieval_parameters"]),
                    tokenizer_identity=str(value["tokenizer_identity"]),
                    top_k=int(value["top_k"]),
                    paragraphs=paragraphs,
                )
            )
    if not records:
        raise ValueError("Retrieval cache is empty.")
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Retrieval cache contains duplicate example IDs.")
    return records


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_unresolved(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in UNRESOLVED)


def deterministic_corpus_object_id(prefix: str, source_identity: Mapping[str, Any]) -> str:
    if not prefix.strip() or not source_identity:
        raise ValueError("Deterministic corpus IDs require a prefix and source identity.")
    return f"{prefix}:{canonical_sha256(source_identity)[:24]}"


def _load_object(path: str | Path, *, yaml_format: bool, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {label}: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) if yaml_format else json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def validate_corpus_manifest(
    manifest: Mapping[str, Any], *, verify_artifacts: bool = True
) -> None:
    if manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Corpus manifest schema_version must be 1.")
    if _is_unresolved(manifest.get("corpus_id")):
        raise ValueError("Corpus manifest is missing corpus identity.")
    if manifest.get("construction_protocol") not in CORPUS_PROTOCOLS:
        raise ValueError("Corpus construction protocol is unresolved or unsupported.")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        _is_unresolved(dataset.get(field))
        for field in ("identity", "version_or_config", "source_splits")
    ):
        raise ValueError("Corpus dataset provenance is incomplete.")
    for field in ("number_of_unique_documents", "number_of_unique_paragraphs"):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Corpus manifest {field} must be a positive integer.")
    for field in ("deduplication_rule", "normalization_rule", "ordering_rule", "tokenizer_identity"):
        if _is_unresolved(manifest.get(field)):
            raise ValueError(f"Corpus manifest {field} is unresolved.")
    construction_inputs = manifest.get("construction_inputs")
    if not isinstance(construction_inputs, list) or not construction_inputs:
        raise ValueError("Corpus construction inputs must be explicit.")
    leaked = sorted(FORBIDDEN_CORPUS_SELECTION_INPUTS.intersection(map(str, construction_inputs)))
    if leaked:
        raise ValueError("Corpus construction cannot use labels or outcomes: " + ", ".join(leaked))
    corpus = manifest.get("corpus_artifact")
    if not isinstance(corpus, Mapping) or not _valid_sha(corpus.get("sha256")) or _is_unresolved(corpus.get("path")):
        raise ValueError("Corpus artifact path/SHA256 is incomplete.")
    index = manifest.get("index")
    if not isinstance(index, Mapping) or _is_unresolved(index.get("type")):
        raise ValueError("Corpus index provenance is incomplete.")
    if index.get("corpus_sha256") != corpus.get("sha256"):
        raise ValueError("Index corpus SHA256 does not match the corpus artifact.")
    if verify_artifacts:
        corpus_path = Path(str(corpus["path"]))
        if not corpus_path.is_file() or _file_sha256(corpus_path) != corpus["sha256"]:
            raise ValueError("Corpus artifact is missing or has the wrong SHA256.")
        index_path = Path(str(index.get("path") or ""))
        if not index_path.is_file() or not _valid_sha(index.get("sha256")):
            raise ValueError("Index artifact path/SHA256 is incomplete.")
        if _file_sha256(index_path) != index["sha256"]:
            raise ValueError("Index artifact has the wrong SHA256.")
        index_payload = _load_object(index_path, yaml_format=False, label="global BM25 index")
        GlobalBM25Index.from_dict(index_payload, expected_corpus_sha256=str(corpus["sha256"]))


def map_support_facts_to_paragraph_ids(
    supporting_facts: Sequence[SupportKey],
    fact_to_paragraph_id: Mapping[SupportKey, str],
) -> tuple[str, ...]:
    missing = [fact for fact in supporting_facts if fact not in fact_to_paragraph_id]
    if missing:
        raise ValueError("Supporting-fact paragraph identity is missing for: " + ", ".join(map(str, missing)))
    return tuple(sorted({fact_to_paragraph_id[fact] for fact in supporting_facts}))


def paragraph_retrieval_metrics(
    retrieved: Sequence[RetrievedParagraph],
    gold_support_paragraph_ids: Sequence[str],
) -> dict[str, float | int | None]:
    if not gold_support_paragraph_ids:
        raise ValueError("Paragraph retrieval metrics require at least one gold support paragraph.")
    retrieved_ids = [paragraph.paragraph_id for paragraph in retrieved]
    if any(identifier is None for identifier in retrieved_ids):
        raise ValueError("Retrieved paragraphs must have explicit paragraph IDs.")
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("Retrieved paragraph IDs must be unique.")
    gold = set(gold_support_paragraph_ids)
    hits = gold.intersection(retrieved_ids)
    reciprocal_rank = 0.0
    for rank, paragraph_id in enumerate(retrieved_ids, start=1):
        if paragraph_id in gold:
            reciprocal_rank = 1.0 / rank
            break
    all_retrieved = float(len(hits) == len(gold))
    return {
        "support_paragraph_hits": len(hits),
        "number_of_gold_support_paragraphs": len(gold),
        "support_paragraph_recall": len(hits) / len(gold),
        "all_support_paragraphs_retrieved_rate": all_retrieved,
        "at_least_one_support_retrieved_rate": float(bool(hits)),
        "mrr": reciprocal_rank,
        "both_gold_supporting_paragraphs_retrieved_rate": all_retrieved if len(gold) == 2 else None,
        "retrieved_candidate_count": len(retrieved_ids),
    }


def aggregate_retrieval_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("Retrieval aggregation requires at least one per-example record.")
    result: dict[str, float | int] = {"N": len(records)}
    for metric in RETRIEVAL_METRIC_NAMES:
        values = [record[metric] for record in records if record.get(metric) is not None]
        if values:
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ValueError(f"Retrieval metric {metric} contains invalid values.")
            result[metric] = sum(values) / len(values)
    latencies = [float(record["retrieval_latency_ms"]) for record in records]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Retrieval latency contains invalid values.")
    result["retrieval_latency_ms"] = sum(latencies) / len(records)
    return result


def aggregate_retrieval_diagnostics(
    per_example_path: str | Path, *, output_path: str | Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with Path(per_example_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Retrieval diagnostic line {line_number} must be an object.")
            records.append(value)
    example_ids = [record.get("example_id") for record in records]
    if any(not isinstance(value, str) or not value for value in example_ids):
        raise ValueError("Retrieval diagnostics contain missing example IDs.")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Retrieval diagnostics contain duplicate example IDs.")
    for field in (
        "retrieval_mode",
        "retrieval_method",
        "retrieval_parameters",
        "tokenizer_identity",
        "corpus_id",
        "corpus_sha256",
        "query_population_sha256",
        "top_k",
        "query_role",
    ):
        if len({canonical_json(record.get(field)) for record in records}) != 1:
            raise ValueError(f"Retrieval diagnostics mix {field} values.")
    aggregate = {
        "retrieval_mode": records[0]["retrieval_mode"],
        "corpus_id": records[0]["corpus_id"],
        "corpus_sha256": records[0]["corpus_sha256"],
        "query_population_sha256": records[0]["query_population_sha256"],
        "query_role": records[0]["query_role"],
        "retrieval_method": records[0]["retrieval_method"],
        "retrieval_parameters": canonical_json(records[0]["retrieval_parameters"]),
        "tokenizer_identity": records[0]["tokenizer_identity"],
        "top_k": records[0]["top_k"],
        **aggregate_retrieval_metrics(records),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    return aggregate


def validate_retrieval_fairness(
    candidates_by_packer: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if len(candidates_by_packer) < 2:
        raise ValueError("Retrieval fairness requires at least two evidence packers.")
    packers = tuple(candidates_by_packer)
    reference = candidates_by_packer[packers[0]]
    reference_signature = [
        (item.get("paragraph_id"), item.get("rank"), item.get("score")) for item in reference
    ]
    for packer in packers[1:]:
        signature = [
            (item.get("paragraph_id"), item.get("rank"), item.get("score"))
            for item in candidates_by_packer[packer]
        ]
        if canonical_json(signature) != canonical_json(reference_signature):
            raise ValueError(
                f"Unfair packing comparison: {packer} received different retrieved candidates/order/scores."
            )


def validate_query_role(
    role: str,
    *,
    query_population_sha256: str,
    expected_population_sha256: str,
    query_population_count: int | None = None,
    expected_population_count: int | None = None,
) -> None:
    if role != GLOBAL_RETRIEVAL_ROLE:
        raise ValueError("Final retrieval evidence requires role='global_retrieval_evaluation'.")
    if query_population_sha256 != expected_population_sha256:
        raise ValueError("Global retrieval query population SHA256 does not match the frozen protocol.")
    if expected_population_count is not None and query_population_count != expected_population_count:
        raise ValueError("Global retrieval query population count does not match the frozen protocol.")


def validate_retrieval_cache_reuse(
    cached: FrozenRetrievedCandidates,
    *,
    query_population_sha256: str,
    corpus_sha256: str,
    retrieval_method: str,
    retrieval_parameters: Mapping[str, Any],
    tokenizer_identity: str,
    top_k: int,
) -> None:
    expected = {
        "query_population_sha256": cached.query_population_sha256,
        "corpus_sha256": cached.corpus_sha256,
        "retrieval_method": cached.retrieval_method,
        "retrieval_parameters": cached.retrieval_parameters,
        "tokenizer_identity": cached.tokenizer_identity,
        "top_k": cached.top_k,
    }
    requested = {
        "query_population_sha256": query_population_sha256,
        "corpus_sha256": corpus_sha256,
        "retrieval_method": retrieval_method,
        "retrieval_parameters": retrieval_parameters,
        "tokenizer_identity": tokenizer_identity,
        "top_k": top_k,
    }
    changed = [
        field for field in expected
        if canonical_json(expected[field]) != canonical_json(requested[field])
    ]
    if changed:
        raise ValueError("Frozen retrieval cache is not reusable; changed: " + ", ".join(changed))


def evaluate_global_retrieval(
    *,
    retriever: GlobalCorpusRetriever,
    queries: Sequence[Mapping[str, Any]],
    query_role: str,
    query_population_sha256: str,
    expected_population_sha256: str,
    top_k: int,
    per_example_output: str | Path,
    aggregate_output: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generator-free retrieval diagnostics; callers supply already-authorized query fixtures/populations."""
    validate_query_role(
        query_role,
        query_population_sha256=query_population_sha256,
        expected_population_sha256=expected_population_sha256,
    )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query in queries:
        example_id = query.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
            raise ValueError("Global retrieval queries require unique non-empty example IDs.")
        seen_ids.add(example_id)
        question = query.get("question")
        gold_ids = query.get("gold_support_paragraph_ids")
        if not isinstance(question, str) or not isinstance(gold_ids, Sequence) or isinstance(gold_ids, str):
            raise ValueError("Global retrieval query is missing question or paragraph-level gold identities.")
        started = time.perf_counter()
        retrieved = retriever.retrieve(question=question, top_k=top_k, query_id=example_id)
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = paragraph_retrieval_metrics(retrieved, [str(value) for value in gold_ids])
        records.append({
            "example_id": example_id,
            "query_role": query_role,
            "corpus_id": retriever.index.corpus_id,
            "corpus_sha256": retriever.index.corpus_sha256,
            "query_population_sha256": query_population_sha256,
            "retrieval_mode": GLOBAL_CORPUS,
            "retrieval_method": "bm25",
            "retrieval_parameters": {"k1": retriever.index.k1, "b": retriever.index.b},
            "tokenizer_identity": retriever.index.tokenizer_identity,
            "top_k": top_k,
            "retrieved_paragraph_ids": [paragraph.paragraph_id for paragraph in retrieved],
            "retrieved_titles": [paragraph.title for paragraph in retrieved],
            "retrieval_scores": [paragraph.score for paragraph in retrieved],
            "gold_support_paragraph_ids": sorted(set(map(str, gold_ids))),
            **metrics,
            "retrieval_latency_ms": latency_ms,
        })
    aggregate = {
        "retrieval_mode": GLOBAL_CORPUS,
        "corpus_id": retriever.index.corpus_id,
        "corpus_sha256": retriever.index.corpus_sha256,
        "query_population_sha256": query_population_sha256,
        "query_role": query_role,
        "retrieval_method": "bm25",
        "retrieval_parameters": canonical_json({"k1": retriever.index.k1, "b": retriever.index.b}),
        "tokenizer_identity": retriever.index.tokenizer_identity,
        "top_k": top_k,
        **aggregate_retrieval_metrics(records),
    }
    per_example_path = Path(per_example_output)
    per_example_path.parent.mkdir(parents=True, exist_ok=True)
    with per_example_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    aggregate_path = Path(aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    return records, aggregate


def build_retrieval_manifest(
    *,
    evaluation_role: str,
    retrieval_mode: str,
    corpus_manifest: Mapping[str, Any],
    query_population: Mapping[str, Any],
    retrieval_parameters: Mapping[str, Any],
    top_k: int,
    tokenizer_identity: str,
    freeze_sha256: str | None,
    code_revision: str | None,
    environment_reference: str | None,
    source_artifacts: Sequence[str | Path],
    aggregate_artifacts: Sequence[str | Path],
    created_at: str,
) -> dict[str, Any]:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError("Unknown retrieval mode.")
    if retrieval_mode == GLOBAL_CORPUS:
        if evaluation_role != GLOBAL_RETRIEVAL_ROLE:
            raise ValueError("Global-corpus manifests require the global_retrieval_evaluation role.")
        validate_corpus_manifest(corpus_manifest)
        if not _valid_sha(freeze_sha256):
            raise ValueError("Global-corpus manifests require the Phase-3 freeze SHA256.")
    if not _valid_sha(query_population.get("sha256")):
        raise ValueError("Retrieval manifest query population SHA256 is missing.")
    population_count = query_population.get("N")
    if not isinstance(population_count, int) or isinstance(population_count, bool) or population_count <= 0:
        raise ValueError("Retrieval manifest query population N must be positive.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("Retrieval manifest top_k must be positive.")
    sources = [Path(path) for path in source_artifacts]
    aggregates = [Path(path) for path in aggregate_artifacts]
    if not sources or not aggregates or any(not path.is_file() for path in (*sources, *aggregates)):
        raise ValueError("Completed retrieval source and aggregate artifacts are required.")
    return {
        "schema_version": RETRIEVAL_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_role": evaluation_role,
        "retrieval_mode": retrieval_mode,
        "corpus": {
            "corpus_id": corpus_manifest["corpus_id"],
            "corpus_sha256": corpus_manifest["corpus_artifact"]["sha256"],
        },
        "query_population": dict(query_population),
        "retrieval_method": "bm25",
        "retrieval_parameters": dict(retrieval_parameters),
        "top_k": top_k,
        "tokenizer_identity": tokenizer_identity,
        "freeze_sha256": freeze_sha256,
        "code_revision": code_revision,
        "environment_reference": environment_reference,
        "source_artifacts": [{"path": str(path), "sha256": _file_sha256(path)} for path in sources],
        "aggregate_artifacts": [{"path": str(path), "sha256": _file_sha256(path)} for path in aggregates],
        "created_at": created_at,
    }


def verify_retrieval_readiness(
    *,
    protocol_path: str | Path,
    freeze_manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> RetrievalReadinessReport:
    checks: list[RetrievalReadinessCheck] = [
        RetrievalReadinessCheck("retrieval modes registered", set(RETRIEVAL_MODES) == {CONTROLLED_CONTEXT, GLOBAL_CORPUS}, "explicit modes"),
        RetrievalReadinessCheck("controlled-context implementation", True, "BM25ParagraphRetriever preserved"),
        RetrievalReadinessCheck("global retriever implementation", True, "precomputed deterministic BM25 index"),
        RetrievalReadinessCheck("metric implementation", True, ", ".join(RETRIEVAL_METRIC_NAMES)),
    ]
    protocol: dict[str, Any] | None = None
    try:
        protocol = _load_object(protocol_path, yaml_format=True, label="global retrieval protocol")
        if protocol.get("schema_version") != 1 or protocol.get("evaluation_role") != GLOBAL_RETRIEVAL_ROLE:
            raise ValueError("Global retrieval protocol role/schema is invalid.")
        retrieval_protocol = protocol.get("retrieval")
        if not isinstance(retrieval_protocol, Mapping):
            raise ValueError("Global retrieval settings are missing.")
        if retrieval_protocol.get("evaluation_mode") != GLOBAL_CORPUS:
            raise ValueError("Global retrieval protocol must explicitly select global_corpus.")
        if protocol.get("tuning_permitted") is not False:
            raise ValueError("Global retrieval protocol must prohibit tuning.")
        corpus_protocol = protocol.get("corpus_protocol")
        if not isinstance(corpus_protocol, Mapping):
            raise ValueError("Global corpus construction protocol is missing.")
        if corpus_protocol.get("construction_protocol") not in CORPUS_PROTOCOLS:
            raise ValueError("Global corpus construction protocol remains unresolved.")
        if any(
            _is_unresolved(corpus_protocol.get(field))
            for field in (
                "dataset_identity",
                "dataset_version_or_config",
                "source_splits",
                "deduplication_rule",
                "normalization_rule",
                "ordering_rule",
            )
        ):
            raise ValueError("Global corpus protocol contains unresolved provenance fields.")
        checks.append(RetrievalReadinessCheck("global corpus protocol", True, "resolved role and mode"))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        checks.append(RetrievalReadinessCheck("global corpus protocol", False, str(exc)))
    identity: dict[str, Any] | None = None
    try:
        freeze = _load_object(freeze_manifest_path, yaml_format=False, label="Phase-3 freeze manifest")
        identity = validate_freeze_boundary(freeze)
        if protocol is not None and protocol.get("freeze_sha256") != identity["freeze_sha256"]:
            raise ValueError("Global retrieval protocol freeze SHA256 is unresolved or mismatched.")
        checks.append(RetrievalReadinessCheck("freeze dependency", True, "frozen method identity validated"))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        checks.append(RetrievalReadinessCheck("freeze dependency", False, str(exc)))
    try:
        if protocol is None:
            raise ValueError("blocked until the global retrieval protocol passes")
        retrieval_protocol = protocol.get("retrieval")
        if not isinstance(retrieval_protocol, Mapping):
            raise ValueError("Global retrieval settings are missing.")
        corpus_path = retrieval_protocol.get("corpus_manifest")
        if _is_unresolved(corpus_path):
            raise ValueError("Real global corpus manifest remains unresolved.")
        corpus = _load_object(str(corpus_path), yaml_format=False, label="global corpus manifest")
        validate_corpus_manifest(corpus)
        checks.append(RetrievalReadinessCheck("corpus manifest", True, "artifact identity and SHA256 validated"))
        checks.append(RetrievalReadinessCheck("index/corpus binding", True, "index bound to corpus SHA256"))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        checks.append(RetrievalReadinessCheck("corpus manifest", False, str(exc)))
        checks.append(RetrievalReadinessCheck("index/corpus binding", False, str(exc)))
    try:
        if protocol is None or identity is None:
            raise ValueError("blocked until protocol and freeze pass")
        population = protocol.get("query_population")
        if not isinstance(population, Mapping):
            raise ValueError("Query population metadata is missing.")
        validate_query_role(
            str(protocol.get("evaluation_role")),
            query_population_sha256=str(population.get("sha256")),
            expected_population_sha256=str(identity["final_split_sha256"]),
            query_population_count=population.get("count"),
            expected_population_count=(
                identity["dataset"].get("final_count")
                if isinstance(identity.get("dataset"), Mapping)
                else None
            ),
        )
        if population.get("ids_file") != "data/splits/final_ids.json":
            raise ValueError("Global retrieval must use the canonical final ID manifest.")
        checks.append(RetrievalReadinessCheck("query-population role", True, "global retrieval role validated"))
        checks.append(RetrievalReadinessCheck("final isolation", True, "population identity only; tuning prohibited"))
    except (ValueError, TypeError) as exc:
        checks.append(RetrievalReadinessCheck("query-population role", False, str(exc)))
        checks.append(RetrievalReadinessCheck("final isolation", False, str(exc)))
    report = RetrievalReadinessReport(tuple(checks))
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report
