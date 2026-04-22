from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


def _load_torch() -> Any:
    import torch

    return torch


def _normalize_device_name(requested: str | None) -> str:
    value = (requested or "auto").lower()
    if value not in {"auto", "xpu", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device '{requested}'. Expected one of: auto, xpu, cuda, mps, cpu.")
    return value


def _backend_available(torch: Any, backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "xpu":
        return bool(getattr(getattr(torch, "xpu", None), "is_available", lambda: False)())
    if backend == "cuda":
        return bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    if backend == "mps":
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(getattr(mps_backend, "is_available", lambda: False)())
    return False


def resolve_device(requested: str | None) -> Any:
    torch = _load_torch()
    normalized = _normalize_device_name(requested)
    if normalized == "auto":
        for backend in ("xpu", "cuda", "mps", "cpu"):
            if _backend_available(torch, backend):
                return torch.device(backend)
        return torch.device("cpu")
    if not _backend_available(torch, normalized):
        raise RuntimeError(f"Requested device '{normalized}' is unavailable in this environment.")
    return torch.device(normalized)


def device_name(device: Any | None = None) -> str:
    if device is None:
        return "auto"
    return getattr(device, "type", str(device))


def pretty_device_name(device: Any) -> str:
    torch = _load_torch()
    backend = device_name(device)
    if backend == "xpu":
        get_name = getattr(getattr(torch, "xpu", None), "get_device_name", None)
        if callable(get_name):
            try:
                return str(get_name(getattr(device, "index", None)))
            except Exception:
                pass
        return "Intel XPU"
    if backend == "cuda":
        get_name = getattr(getattr(torch, "cuda", None), "get_device_name", None)
        if callable(get_name):
            try:
                return str(get_name(getattr(device, "index", None)))
            except Exception:
                pass
        return "CUDA GPU"
    if backend == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def get_default_dtype_for_device(device: Any) -> Any:
    torch = _load_torch()
    if device_name(device) in {"xpu", "cuda"}:
        return torch.float16
    return torch.float32


def dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def resolve_dtype(requested: str | None, device: Any) -> Any:
    torch = _load_torch()
    normalized = (requested or "auto").lower()
    if normalized == "auto":
        return get_default_dtype_for_device(device)
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        if device_name(device) == "cpu":
            raise RuntimeError("float16 is not supported for CPU execution in this project. Use --dtype auto or --dtype float32.")
        return torch.float16
    raise ValueError(f"Unsupported dtype '{requested}'. Expected one of: auto, float32, float16.")


def is_gpu_device(device: Any | None) -> bool:
    return device is not None and device_name(device) in {"xpu", "cuda", "mps"}


def move_batch_to_device(batch: Any, device: Any) -> Any:
    torch = _load_torch()
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def synchronize_device(device: Any | None) -> None:
    if device is None:
        return
    torch = _load_torch()
    backend = device_name(device)
    if backend == "xpu" and _backend_available(torch, "xpu"):
        torch.xpu.synchronize()
    elif backend == "cuda" and _backend_available(torch, "cuda"):
        torch.cuda.synchronize()


def elapsed_time_ms(start_time: float, device: Any | None = None) -> float:
    synchronize_device(device)
    return (time.perf_counter() - start_time) * 1000.0
