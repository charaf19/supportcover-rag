from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from supportcover_rag.config import ExternalCompressorConfig
from supportcover_rag.external_baselines.base import (
    EvidenceCompressor,
    ExternalCompressorMetadata,
    PublicationEvidenceCompressor,
)


def validate_external_compressor_metadata(
    compressor: EvidenceCompressor,
    config: ExternalCompressorConfig,
) -> ExternalCompressorMetadata:
    if not isinstance(compressor, PublicationEvidenceCompressor):
        raise TypeError("Configured external compressor must expose publication metadata.")
    metadata = compressor.metadata
    if not isinstance(metadata, ExternalCompressorMetadata):
        raise TypeError("External compressor metadata must be ExternalCompressorMetadata.")
    expected = ExternalCompressorMetadata(
        implementation_id=config.implementation_id,
        version=config.version,
        revision=config.revision,
        preserves_support_keys=config.preserves_support_keys,
    )
    if metadata != expected:
        raise ValueError("Configured external compressor provenance does not match adapter metadata.")
    return metadata


def load_configured_external_compressor(config: ExternalCompressorConfig) -> EvidenceCompressor:
    """Load one explicitly configured adapter; never select or substitute an implementation."""
    if not config.enabled:
        raise RuntimeError("external_compressor is selected but external_compressor.enabled is false.")
    required = {
        "adapter": config.adapter,
        "implementation_id": config.implementation_id,
        "version": config.version,
        "revision": config.revision,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        raise ValueError("External compressor configuration is missing: " + ", ".join(missing))
    if ":" not in config.adapter:
        raise ValueError("external_compressor.adapter must use 'module:factory' syntax.")
    module_name, factory_name = config.adapter.split(":", maxsplit=1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Configured external compressor module is unavailable: {module_name}") from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RuntimeError(f"Configured external compressor factory is unavailable: {config.adapter}")
    compressor = _call_factory(factory, config.parameters)
    if not isinstance(compressor, EvidenceCompressor):
        raise TypeError("Configured external compressor factory did not return an EvidenceCompressor.")
    validate_external_compressor_metadata(compressor, config)
    return compressor


def _call_factory(factory: Callable[..., Any], parameters: dict[str, Any]) -> Any:
    try:
        return factory(**parameters)
    except TypeError as exc:
        raise TypeError("External compressor factory rejected the configured parameters.") from exc
