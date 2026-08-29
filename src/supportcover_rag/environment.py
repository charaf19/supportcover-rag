from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping
from typing import Any, Callable

import psutil


def _safe_call(function: Callable[[], Any], default: Any = None) -> Any:
    try:
        return function()
    except Exception:
        return default


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_environment_manifest(
    *,
    model: str | None = None,
    dtype: str | None = None,
    batch_settings: Mapping[str, Any] | None = None,
    code_revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    virtual_memory = _safe_call(psutil.virtual_memory)
    manifest: dict[str, Any] = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "identity": platform.processor() or platform.machine() or None,
            "physical_cores": _safe_call(lambda: psutil.cpu_count(logical=False)),
            "logical_cores": _safe_call(lambda: psutil.cpu_count(logical=True)),
        },
        "memory": {
            "total_bytes": getattr(virtual_memory, "total", None),
        },
        "packages": {
            distribution: _package_version(distribution)
            for distribution in (
                "supportcover-rag",
                "numpy",
                "psutil",
                "PyYAML",
                "datasets",
                "torch",
                "transformers",
            )
        },
        "accelerator": {
            "available": False,
            "backend": None,
            "device_name": None,
            "device_memory_bytes": None,
        },
        "run": {
            "model": model,
            "dtype": dtype,
            "batch_settings": dict(batch_settings or {}),
            "code_revision": code_revision,
        },
        "metadata": dict(metadata or {}),
    }

    try:
        import torch

        cuda_available = bool(_safe_call(torch.cuda.is_available, False))
        if cuda_available:
            manifest["accelerator"] = {
                "available": True,
                "backend": "cuda",
                "device_name": _safe_call(lambda: torch.cuda.get_device_name(0)),
                "device_memory_bytes": _safe_call(lambda: int(torch.cuda.get_device_properties(0).total_memory)),
                "compute_capability": _safe_call(lambda: list(torch.cuda.get_device_capability(0))),
                "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
                "cudnn_version": _safe_call(torch.backends.cudnn.version),
            }
        else:
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            mps_available = bool(_safe_call(mps_backend.is_available, False)) if mps_backend is not None else False
            if mps_available:
                manifest["accelerator"] = {
                    "available": True,
                    "backend": "mps",
                    "device_name": "Apple Metal Performance Shaders",
                    "device_memory_bytes": None,
                }
    except Exception:
        pass

    try:
        import transformers

        manifest["packages"]["transformers"] = getattr(transformers, "__version__", manifest["packages"]["transformers"])
    except Exception:
        pass
    return manifest
