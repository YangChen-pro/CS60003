"""Configuration helpers for HW3 Task1 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "task1_ai_generated_smoke",
        "seed": 42,
        "output_root": "hw3/task1/outputs",
    },
    "data": {
        "source_root": "hw3/assets/ai_generated_test",
        "object_a_dir": "hw3/assets/ai_generated_test/object_a_multiview",
        "object_c_image": "hw3/assets/ai_generated_test/object_c_single/object_c_single_front.png",
        "min_image_size": 1024,
    },
    "task1": {
        "stage": "smoke_assets",
        "object_a_expected_yaws": ["000", "045", "090", "135", "180", "225", "270", "315"],
        "final_submission_assets": False,
        "note": "AI-generated placeholders only.",
    },
    "external_tools": {
        "optional": ["colmap", "ffmpeg", "blender", "nvidia-smi"],
    },
    "logging": {
        "swanlab": {
            "enabled": False,
            "project": "cs60003-hw3-task1",
            "mode": "cloud",
            "group": "smoke",
            "tags": ["hw3", "task1", "smoke"],
        }
    },
    "formal_chain": {
        "object_a_sample_size": 176,
        "object_c_sample_size": 176,
        "object_b_prompt": "a small bioluminescent purple crystal mushroom with a green stem",
        "object_b_radial_steps": 48,
        "object_b_height_steps": 28,
        "background_grid": 70,
        "placements": {
            "object_a": {"scale": 0.82, "translate": [-1.05, -0.10, 0.05], "yaw_degrees": 8.0},
            "object_b": {"scale": 1.00, "translate": [0.08, -0.38, -0.22], "yaw_degrees": 0.0},
            "object_c": {"scale": 0.86, "translate": [1.15, -0.12, 0.05], "yaw_degrees": -10.0},
        },
        "render": {
            "frame_count": 24,
            "width": 960,
            "height": 640,
            "camera_distance": 5.2,
            "focal": 430.0,
        },
    },
    "real_chain": {
        "execution": {"mode": "plan"},
        "data": {
            "object_a_images": "hw3/assets/object_a_multiview",
            "object_a_video": "",
            "object_c_image": "hw3/assets/object_c_single/object_c_single_front.png",
            "background_images": "hw3/assets/background_scene/images",
            "background_video": "",
            "min_object_a_images": 40,
            "min_background_images": 80,
        },
        "tools": {
            "required_cli": ["ns-process-data", "ns-train", "ns-export", "ns-eval", "colmap", "ffmpeg", "blender"],
            "threestudio_launch": "hw3/task1/external/threestudio/launch.py",
            "triposr_run": "hw3/task1/external/TripoSR/run.py",
            "blender_renderer": "hw3/task1/scripts/render_real_chain_blender.py",
        },
        "quality": {
            "splatfacto_iterations": 30000,
            "cull_alpha_thresh": 0.005,
        },
        "object_b": {
            "prompt": "a small bioluminescent purple crystal mushroom with a green stem",
            "max_steps": 15000,
        },
        "object_c": {
            "texture_resolution": 2048,
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and fill missing values from defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    _validate_config(config)
    return config


def repo_root_from_task_file() -> Path:
    """Return repository root from this package location."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a path relative to repository root unless absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root or repo_root_from_task_file()) / path


def resolve_paths(config: dict[str, Any], output_root: str | None = None) -> None:
    """Resolve configured paths in-place relative to repository root."""
    repo_root = repo_root_from_task_file()
    config["data"]["source_root"] = str(resolve_repo_path(config["data"]["source_root"]))
    config["data"]["object_a_dir"] = str(resolve_repo_path(config["data"]["object_a_dir"]))
    config["data"]["object_c_image"] = str(resolve_repo_path(config["data"]["object_c_image"]))
    if output_root is not None:
        config["experiment"]["output_root"] = output_root
    config["experiment"]["output_root"] = str(resolve_repo_path(config["experiment"]["output_root"]))
    report_dir = config["experiment"].get("report_assets_dir", "")
    if report_dir:
        config["experiment"]["report_assets_dir"] = str(resolve_repo_path(report_dir, repo_root))
    if "real_chain" in config:
        real_data = config["real_chain"]["data"]
        for key in ["object_a_images", "object_c_image", "background_images"]:
            real_data[key] = str(resolve_repo_path(real_data[key], repo_root))
        for key in ["object_a_video", "background_video"]:
            if real_data.get(key):
                real_data[key] = str(resolve_repo_path(real_data[key], repo_root))
        tools = config["real_chain"]["tools"]
        for key in ["threestudio_launch", "triposr_run", "blender_renderer"]:
            tools[key] = str(resolve_repo_path(tools[key], repo_root))


def _validate_config(config: dict[str, Any]) -> None:
    stage = str(config.get("task1", {}).get("stage", ""))
    if stage not in {"smoke_assets", "formal_ai_chain", "real_high_quality"}:
        raise ValueError(f"Unsupported Task1 stage for current scaffold: {stage}")
    yaws = config.get("task1", {}).get("object_a_expected_yaws")
    if not isinstance(yaws, list) or len(yaws) != 8:
        raise ValueError("task1.object_a_expected_yaws must contain 8 yaw labels.")
    if int(config.get("data", {}).get("min_image_size", 0)) < 256:
        raise ValueError("data.min_image_size is unexpectedly small.")
