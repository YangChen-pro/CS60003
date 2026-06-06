"""Run threestudio with SwanLab sync for its WandB scalar logger."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import sysconfig
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments and preserve the threestudio CLI after `--`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--mode", default="cloud")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("threestudio_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.threestudio_args and args.threestudio_args[0] == "--":
        args.threestudio_args = args.threestudio_args[1:]
    if not args.threestudio_args:
        raise ValueError("Missing threestudio command after --.")
    return args


def main() -> None:
    """Patch WandB logging into SwanLab, then run threestudio launch.py."""
    args = parse_args()
    _load_env_file(args.env_file)
    api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is required for training SwanLab logging.")
    _prepare_cuda_extension_env(args.logdir)
    _patch_trusted_torch_load()

    import swanlab

    swanlab.login(api_key=api_key)
    swanlab.sync_wandb(mode=args.mode, wandb_run=False, logdir=args.logdir)
    launch_path = Path(args.launch)
    launch_root = str(launch_path.parent.resolve())
    if launch_root not in sys.path:
        sys.path.insert(0, launch_root)
    sys.argv = [str(launch_path), *args.threestudio_args]
    runpy.run_path(str(launch_path), run_name="__main__")


def _load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _prepare_cuda_extension_env(logdir: str | Path) -> None:
    """Expose pip-installed CUDA headers/libs for threestudio JIT extensions."""
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
    extension_dir = _torch_extensions_dir(logdir)
    extension_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_dir))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _patch_trusted_torch_load() -> None:
    """Keep Lightning resume compatible with PyTorch 2.6+ for local HW3 checkpoints."""
    enabled = os.environ.get("HW3_TRUSTED_TORCH_LOAD_WEIGHTS_ONLY_FALSE", "1").strip().lower()
    if enabled in {"0", "false", "no"}:
        return
    import torch

    original_load = torch.load
    if getattr(original_load, "_hw3_trusted_patch", False):
        return

    def load_with_trusted_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_trusted_default._hw3_trusted_patch = True
    torch.load = load_with_trusted_default


def _prepend_path_env(name: str, values: list[str]) -> None:
    existing = [value for value in os.environ.get(name, "").split(":") if value]
    merged: list[str] = []
    for value in [*values, *existing]:
        if value and value not in merged:
            merged.append(value)
    if merged:
        os.environ[name] = ":".join(merged)


def _torch_extensions_dir(logdir: str | Path) -> Path:
    log_path = Path(logdir).resolve()
    if len(log_path.parents) >= 2:
        return log_path.parents[1] / "torch_extensions"
    return Path.home() / ".cache" / "torch_extensions_hw3"


if __name__ == "__main__":
    main()
