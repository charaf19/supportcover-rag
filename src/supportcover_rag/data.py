from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from supportcover_rag.io_utils import read_jsonl, write_jsonl, ensure_dir
from supportcover_rag.types import HotpotExample, Paragraph

LOGGER = logging.getLogger(__name__)


def acquire_hotpotqa(dataset_path: str, dataset_config: str, splits: list[str], output_dir: str | Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for data acquisition. Install project dependencies first.") from exc

    target_dir = ensure_dir(output_dir)
    dataset = load_dataset(dataset_path, dataset_config)

    for split in splits:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset '{dataset_path}/{dataset_config}'.")
        rows = [dict(row) for row in dataset[split]]
        path = target_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        LOGGER.info("Wrote raw split '%s' to %s", split, path)


def _normalize_raw_record(raw: dict) -> dict:
    context = []
    for title, sentences in zip(raw["context"]["title"], raw["context"]["sentences"], strict=True):
        context.append({"title": title, "sentences": [sentence.strip() for sentence in sentences if sentence.strip()]})

    supporting_facts = [
        {"title": title, "sent_id": int(sent_id)}
        for title, sent_id in zip(raw["supporting_facts"]["title"], raw["supporting_facts"]["sent_id"], strict=True)
    ]

    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw["type"],
        "level": raw["level"],
        "context": context,
        "supporting_facts": supporting_facts,
    }


def preprocess_raw_split(raw_path: str | Path, processed_path: str | Path, limit: int | None = None) -> None:
    raw_rows = read_jsonl(raw_path)
    normalized = []
    for index, row in enumerate(raw_rows):
        if limit is not None and index >= limit:
            break
        normalized.append(_normalize_raw_record(row))
    write_jsonl(processed_path, normalized)
    LOGGER.info("Preprocessed %d records from %s to %s", len(normalized), raw_path, processed_path)


def load_examples(processed_path: str | Path, limit: int | None = None) -> list[HotpotExample]:
    rows = read_jsonl(processed_path)
    examples: list[HotpotExample] = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        paragraphs = [Paragraph(title=paragraph["title"], sentences=list(paragraph["sentences"])) for paragraph in row["context"]]
        supporting_facts = [(fact["title"], int(fact["sent_id"])) for fact in row["supporting_facts"]]
        examples.append(
            HotpotExample(
                example_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                qtype=row["type"],
                level=row["level"],
                context=paragraphs,
                supporting_facts=supporting_facts,
            )
        )
    return examples


def validate_processed_rows(rows: Iterable[dict]) -> None:
    for row in rows:
        if not row.get("question"):
            raise ValueError(f"Missing question in record {row.get('id')}")
        if not row.get("answer"):
            raise ValueError(f"Missing answer in record {row.get('id')}")
        if not row.get("context"):
            raise ValueError(f"Missing context in record {row.get('id')}")
        titles = {paragraph["title"] for paragraph in row["context"]}
        for fact in row.get("supporting_facts", []):
            if fact["title"] not in titles:
                raise ValueError(f"Supporting fact title {fact['title']!r} not present in context for record {row.get('id')}")
