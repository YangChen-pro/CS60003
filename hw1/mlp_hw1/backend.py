"""Backend helpers for NumPy/CuPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


@dataclass(frozen=True)
class Backend:
    """Resolved array backend."""

    name: str
    xp: Any


def resolve_backend(name: str) -> Backend:
    """Resolve the requested array backend."""
    normalized = name.lower()
    if normalized == "auto":
        return resolve_backend("cupy" if cp is not None else "numpy")
    if normalized == "cupy":
        if cp is None:
            raise RuntimeError("未检测到 CuPy，请切换到 numpy 后端或安装 CuPy。")
        return Backend(name="cupy", xp=cp)
    if normalized == "numpy":
        return Backend(name="numpy", xp=np)
    raise ValueError(f"不支持的后端: {name}")


def seed_everything(seed: int) -> None:
    """Seed all available random generators."""
    np.random.seed(seed)
    if cp is not None:
        cp.random.seed(seed)


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array from NumPy/CuPy to NumPy."""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
