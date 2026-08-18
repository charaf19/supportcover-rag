from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from supportcover_rag.benchmark import (
    BenchmarkResult,
    benchmark_callable,
    build_raw_latency_rows,
    build_summary_row,
    summarize_latency_samples,
)
from supportcover_rag.resource_monitor import PeakRSSMonitor


@dataclass
class _MemoryInfo:
    rss: int


class _FakeProcess:
    def __init__(self, rss_values: list[int]) -> None:
        self._rss_values = iter(rss_values)
        self._last = rss_values[-1]

    def memory_info(self) -> _MemoryInfo:
        self._last = next(self._rss_values, self._last)
        return _MemoryInfo(self._last)


class _FakeMonitor:
    def __init__(self, peak_rss_mb: float) -> None:
        self.peak_rss_mb = peak_rss_mb
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> float:
        self.stops += 1
        return self.peak_rss_mb


def test_benchmark_summary_for_fixed_samples() -> None:
    summary = summarize_latency_samples([1.0, 2.0, 3.0, 4.0])

    assert summary.count == 4
    assert summary.mean_ms == 2.5
    assert summary.std_ms == pytest.approx(math.sqrt(1.25))
    assert summary.p50_ms == 2.5
    assert summary.p95_ms == pytest.approx(3.85)
    assert summary.min_ms == 1.0
    assert summary.max_ms == 4.0


def test_raw_and_summary_rows_preserve_samples() -> None:
    samples = (2.0, 4.0)
    result = BenchmarkResult(samples, summarize_latency_samples(samples), peak_rss_mb=12.0)

    raw_rows = build_raw_latency_rows(result, benchmark_name="synthetic")
    summary_row = build_summary_row(result, benchmark_name="synthetic")

    assert [row["latency_ms"] for row in raw_rows] == [2.0, 4.0]
    assert [row["sample_index"] for row in raw_rows] == [0, 1]
    assert summary_row["peak_rss_mb"] == 12.0
    assert summary_row["mean_ms"] == 3.0


def test_peak_rss_monitor_lifecycle_is_repeatable_and_guarded() -> None:
    megabyte = 1024 * 1024
    monitor = PeakRSSMonitor(
        polling_interval_seconds=1.0,
        process=_FakeProcess([10 * megabyte, 20 * megabyte, 5 * megabyte, 7 * megabyte]),
    )

    monitor.start()
    with pytest.raises(RuntimeError, match="already running"):
        monitor.start()
    assert monitor.stop() == 20.0
    assert monitor.stop() == 20.0

    monitor.start()
    assert monitor.stop() == 7.0


def test_benchmark_callable_handles_optional_rss_safely() -> None:
    calls: list[int] = []
    clock_values = iter([0.0, 0.001, 0.001, 0.003])

    without_rss = benchmark_callable(
        lambda: calls.append(1),
        warmup_count=1,
        repetition_count=2,
        clock=lambda: next(clock_values),
    )

    assert len(calls) == 3
    assert without_rss.latency_samples_ms == pytest.approx((1.0, 2.0))
    assert without_rss.peak_rss_mb is None

    fake_monitor = _FakeMonitor(9.5)
    with_rss = benchmark_callable(
        lambda: None,
        repetition_count=1,
        rss_monitor=fake_monitor,
        clock=iter([0.0, 0.001]).__next__,
    )
    assert with_rss.peak_rss_mb == 9.5
    assert fake_monitor.starts == 1
    assert fake_monitor.stops == 1
