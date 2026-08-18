from __future__ import annotations

import threading
from typing import Protocol

import psutil


class MemoryInfo(Protocol):
    rss: int


class RSSProcess(Protocol):
    def memory_info(self) -> MemoryInfo:
        ...


class PeakRSSMonitor:
    def __init__(
        self,
        *,
        polling_interval_seconds: float = 0.010,
        process: RSSProcess | None = None,
    ) -> None:
        if polling_interval_seconds <= 0.0:
            raise ValueError("polling_interval_seconds must be positive.")
        self.polling_interval_seconds = polling_interval_seconds
        self._process = process if process is not None else psutil.Process()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0
        self._monitor_error: Exception | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None

    @property
    def peak_rss_mb(self) -> float:
        with self._lock:
            return self._peak_rss_bytes / (1024.0 * 1024.0)

    def _sample(self) -> None:
        rss_bytes = int(self._process.memory_info().rss)
        with self._lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)

    def _monitor(self) -> None:
        try:
            while not self._stop_event.wait(self.polling_interval_seconds):
                self._sample()
        except Exception as exc:
            with self._lock:
                self._monitor_error = exc
            self._stop_event.set()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("PeakRSSMonitor is already running.")
            self._peak_rss_bytes = 0
            self._monitor_error = None
            self._stop_event.clear()
            self._peak_rss_bytes = int(self._process.memory_info().rss)
            thread = threading.Thread(target=self._monitor, name="peak-rss-monitor", daemon=True)
            self._thread = thread
            thread.start()

    def stop(self) -> float:
        with self._lock:
            thread = self._thread
            if thread is None:
                return self._peak_rss_bytes / (1024.0 * 1024.0)
            self._stop_event.set()
        thread.join()
        try:
            self._sample()
        except Exception as exc:
            with self._lock:
                if self._monitor_error is None:
                    self._monitor_error = exc
        with self._lock:
            self._thread = None
            error = self._monitor_error
            peak_rss_mb = self._peak_rss_bytes / (1024.0 * 1024.0)
        if error is not None:
            raise RuntimeError(f"Peak RSS monitoring failed: {error}") from error
        return peak_rss_mb

    def __enter__(self) -> PeakRSSMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
