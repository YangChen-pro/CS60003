"""CuPy backend helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import cupy as cp


def get_array_module() -> Any:
    """Return the only supported array backend."""
    return cp


def seed_everything(seed: int) -> None:
    """Seed CPU/GPU random generators used by the project."""
    np.random.seed(seed)
    cp.random.seed(seed)


def to_numpy(array: Any) -> np.ndarray:
    """Convert a CuPy array to NumPy for reporting or visualization."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
