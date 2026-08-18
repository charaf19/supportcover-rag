from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from supportcover_rag.types import HotpotExample


@runtime_checkable
class DatasetAdapter(Protocol):
    """Normalize an already-loaded multi-hop QA record into the canonical example type."""

    dataset_name: str

    def normalize_record(self, record: Mapping[str, Any]) -> HotpotExample:
        ...
