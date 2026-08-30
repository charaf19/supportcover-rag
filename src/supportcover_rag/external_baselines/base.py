from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from supportcover_rag.types import PackedEvidence, RetrievedParagraph


@dataclass(frozen=True, slots=True)
class ExternalCompressorMetadata:
    implementation_id: str
    version: str
    revision: str
    preserves_support_keys: bool


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


@runtime_checkable
class PublicationEvidenceCompressor(EvidenceCompressor, Protocol):
    """Publication adapter contract with immutable implementation provenance."""

    @property
    def metadata(self) -> ExternalCompressorMetadata:
        ...
