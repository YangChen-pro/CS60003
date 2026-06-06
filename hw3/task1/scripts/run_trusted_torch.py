"""Run a Python console script while allowing trusted HW3 checkpoints to load."""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
import sysconfig
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the wrapped command and its arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """Patch torch.load for trusted local HW3 checkpoints, then run command."""
    args = parse_args()
    _prepare_cuda_extension_env()
    _patch_trusted_torch_load()
    command_path = _resolve_command(args.command)
    command_root = str(Path(command_path).resolve().parent)
    if command_root not in sys.path:
        sys.path.insert(0, command_root)
    sys.argv = [command_path, *args.args]
    runpy.run_path(command_path, run_name="__main__")


def _resolve_command(command: str) -> str:
    path = Path(command)
    if path.is_file():
        return str(path)
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Unable to resolve command: {command}")


def _patch_trusted_torch_load() -> None:
    import torch

    original_load = torch.load
    if getattr(original_load, "_hw3_trusted_patch", False):
        return

    def load_with_trusted_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_trusted_default._hw3_trusted_patch = True
    torch.load = load_with_trusted_default


def _prepare_cuda_extension_env() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = purelib / "nvidia"
    include_dirs = sorted(str(path) for path in nvidia_root.glob("*/include") if path.is_dir())
    lib_dirs = sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        include_dirs.insert(0, str(Path(conda_prefix) / "targets" / "x86_64-linux" / "include"))
        include_dirs.insert(0, str(Path(conda_prefix) / "include"))
        lib_dirs.insert(0, str(Path(conda_prefix) / "lib"))
        os.environ.setdefault("CUDA_HOME", conda_prefix)
    _prepend_path_env("CPATH", include_dirs)
    _prepend_path_env("CPLUS_INCLUDE_PATH", include_dirs)
    _prepend_path_env("LIBRARY_PATH", lib_dirs)
    _prepend_path_env("LD_LIBRARY_PATH", lib_dirs)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _prepend_path_env(name: str, values: list[str]) -> None:
    existing = [value for value in os.environ.get(name, "").split(":") if value]
    merged: list[str] = []
    for value in [*values, *existing]:
        if value and value not in merged:
            merged.append(value)
    if merged:
        os.environ[name] = ":".join(merged)


if __name__ == "__main__":
    main()
