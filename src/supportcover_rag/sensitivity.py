from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from supportcover_rag.config import SensitivityConfig, SupportCoverConfig


_FACTOR_FIELDS = (
    ("beta", "beta_coverage"),
    ("title", "title_bonus"),
    ("delta", "delta_token_cost"),
    ("gamma", "gamma_redundancy"),
)
_REQUIRED_SENSITIVITY_ROLE = "development"


@dataclass(frozen=True, slots=True)
class OFATConfigDescriptor:
    split_role: str
    factor: str
    config_field: str
    value: float
    supportcover: SupportCoverConfig


@dataclass(frozen=True, slots=True)
class OFATResultDescriptor:
    split_role: str
    factor: str
    config_field: str
    value: float
    metrics: dict[str, Any]


def validate_sensitivity_role(split_role: str) -> str:
    normalized = split_role.strip().lower()
    if normalized != _REQUIRED_SENSITIVITY_ROLE:
        rendered = normalized or "<empty>"
        raise ValueError(
            f"Sensitivity analysis requires split role 'development'; received '{rendered}'."
        )
    return normalized


def build_ofat_descriptors(
    base: SupportCoverConfig,
    sensitivity: SensitivityConfig,
    *,
    split_role: str,
) -> list[OFATConfigDescriptor]:
    """Construct deterministic one-factor-at-a-time configurations without executing them."""
    normalized_role = validate_sensitivity_role(split_role)
    descriptors: list[OFATConfigDescriptor] = []
    for factor, config_field in _FACTOR_FIELDS:
        for raw_value in getattr(sensitivity, factor):
            value = float(raw_value)
            descriptors.append(
                OFATConfigDescriptor(
                    split_role=normalized_role,
                    factor=factor,
                    config_field=config_field,
                    value=value,
                    supportcover=replace(base, **{config_field: value}),
                )
            )
    return descriptors


def build_ofat_result_descriptor(
    descriptor: OFATConfigDescriptor,
    metrics: Mapping[str, Any],
) -> OFATResultDescriptor:
    validate_sensitivity_role(descriptor.split_role)
    return OFATResultDescriptor(
        split_role=descriptor.split_role,
        factor=descriptor.factor,
        config_field=descriptor.config_field,
        value=descriptor.value,
        metrics=dict(metrics),
    )
