from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from supportcover_rag.types import PackedEvidence, RetrievedParagraph


@runtime_checkable
class EvidenceCompressor(Protocol):
    """Contract for external evidence compressors without prescribing an implementation."""

    def compress(
        self,
        *,
        question: str,
        retrieved_paragraphs: Sequence[RetrievedParagraph],
        token_budget: int,
    ) -> PackedEvidence:
        """Return evidence compatible with the existing packing and evaluation pipeline."""
        ...
