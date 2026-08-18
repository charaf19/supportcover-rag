from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


T = TypeVar("T")


class RSSMonitor(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> float:
        ...


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    latency_samples_ms: tuple[float, ...]
    summary: LatencySummary
    peak_rss_mb: float | None


def _percentile(samples: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        raise ValueError("At least one latency sample is required.")
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def summarize_latency_samples(samples_ms: Sequence[float]) -> LatencySummary:
    samples = tuple(float(sample) for sample in samples_ms)
    if not samples:
        raise ValueError("At least one latency sample is required.")
    if any(sample < 0.0 for sample in samples):
        raise ValueError("Latency samples must be non-negative.")
    return LatencySummary(
        count=len(samples),
        mean_ms=statistics.fmean(samples),
        std_ms=statistics.pstdev(samples),
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def benchmark_callable(
    function: Callable[..., T],
    *args: Any,
    warmup_count: int = 0,
    repetition_count: int = 1,
    rss_monitor: RSSMonitor | None = None,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: Any,
) -> BenchmarkResult:
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if repetition_count <= 0:
        raise ValueError("repetition_count must be positive.")

    for _ in range(warmup_count):
        function(*args, **kwargs)

    samples_ms: list[float] = []
    peak_rss_mb: float | None = None
    if rss_monitor is not None:
        rss_monitor.start()
    try:
        for _ in range(repetition_count):
            started_at = clock()
            function(*args, **kwargs)
            elapsed_ms = (clock() - started_at) * 1000.0
            samples_ms.append(elapsed_ms)
    finally:
        if rss_monitor is not None:
            peak_rss_mb = rss_monitor.stop()

    return BenchmarkResult(
        latency_samples_ms=tuple(samples_ms),
        summary=summarize_latency_samples(samples_ms),
        peak_rss_mb=peak_rss_mb,
    )


def _merge_metadata(row: dict[str, Any], metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return row
    conflicts = sorted(set(row) & set(metadata))
    if conflicts:
        raise ValueError(f"Benchmark metadata conflicts with reserved fields: {', '.join(conflicts)}")
    return {**dict(metadata), **row}


def build_raw_latency_rows(
    result: BenchmarkResult,
    *,
    benchmark_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        _merge_metadata(
            {
                "benchmark": benchmark_name,
                "sample_index": index,
                "latency_ms": latency_ms,
            },
            metadata,
        )
        for index, latency_ms in enumerate(result.latency_samples_ms)
    ]


def build_summary_row(
    result: BenchmarkResult,
    *,
    benchmark_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = result.summary
    return _merge_metadata(
        {
            "benchmark": benchmark_name,
            "count": summary.count,
            "mean_ms": summary.mean_ms,
            "std_ms": summary.std_ms,
            "p50_ms": summary.p50_ms,
            "p95_ms": summary.p95_ms,
            "min_ms": summary.min_ms,
            "max_ms": summary.max_ms,
            "peak_rss_mb": result.peak_rss_mb,
        },
        metadata,
    )
