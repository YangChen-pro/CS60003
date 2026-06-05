"""High-quality external-tool chain for HW3 Task1.

This module does not replace COLMAP, Nerfstudio, threestudio, TripoSR, or
Blender. It provides a reproducible orchestration layer so replacing
`hw3/assets` with real captures can drive the real tools directly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from task1_3dgs_aigc.utils import copy_source_config, make_run_dir, save_json


def run_real_chain(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare or run the high-quality Task1 external-tool chain."""
    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    scripts_dir = run_dir / "scripts"
    logs_dir = run_dir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    copy_source_config(config["config_path"], run_dir)
    save_json(run_dir / "config.json", config)

    input_issues = _validate_real_inputs(config, strict=config["real_chain"]["execution"]["mode"] == "run")
    scripts = _write_scripts(config, run_dir, scripts_dir)
    summary = {
        "status": _initial_status(config, input_issues),
        "stage": "real_high_quality",
        "run_dir": run_dir.as_posix(),
        "script_count": len(scripts),
        "scripts": [path.as_posix() for path in scripts],
        "input_issues": input_issues,
        "expected_outputs": _expected_outputs(run_dir),
        "input_policy": "Replace hw3/assets real input directories; do not edit code.",
    }
    save_json(run_dir / "summary.json", summary)

    if config["real_chain"]["execution"]["mode"] == "run":
        _run_scripts(scripts, logs_dir)
        summary["status"] = "PASS"
        _collect_outputs(run_dir)
        save_json(run_dir / "summary.json", summary)
    return summary


def _validate_real_inputs(config: dict[str, Any], *, strict: bool) -> list[str]:
    data = config["real_chain"]["data"]
    issues: list[str] = []
    object_a_images = Path(data["object_a_images"])
    object_a_video = str(data.get("object_a_video", "")).strip()
    if not object_a_images.is_dir() and not object_a_video:
        issues.append(f"Missing object A source: {object_a_images} or real_chain.data.object_a_video")
    if object_a_images.is_dir() and not _image_files(object_a_images):
        issues.append(f"Object A image directory has no PNG/JPG files: {object_a_images}")

    object_c_image = Path(data["object_c_image"])
    if not object_c_image.is_file():
        issues.append(f"Missing object C image: {object_c_image}")

    background_images = Path(data["background_images"])
    background_video = str(data.get("background_video", "")).strip()
    if not background_images.is_dir() and not background_video:
        issues.append(f"Missing background 3DGS source: {background_images} or real_chain.data.background_video")
    if background_images.is_dir() and not _image_files(background_images):
        issues.append(f"Background image directory has no PNG/JPG files: {background_images}")
    if strict and issues:
        raise ValueError("; ".join(issues))
    return issues


def _initial_status(config: dict[str, Any], input_issues: list[str]) -> str:
    if config["real_chain"]["execution"]["mode"] == "run":
        return "RUNNING"
    return "READY" if not input_issues else "NEEDS_INPUTS"


def _write_scripts(config: dict[str, Any], run_dir: Path, scripts_dir: Path) -> list[Path]:
    scripts = [
        _write_script(scripts_dir / "00_check_tools.sh", _check_tools_script(config)),
        _write_script(scripts_dir / "01_object_a_splatfacto.sh", _splatfacto_script(config, run_dir, "object_a")),
        _write_script(scripts_dir / "02_background_splatfacto.sh", _splatfacto_script(config, run_dir, "background")),
        _write_script(scripts_dir / "03_object_b_threestudio.sh", _object_b_script(config, run_dir)),
        _write_script(scripts_dir / "04_object_c_triposr.sh", _object_c_script(config, run_dir)),
        _write_script(scripts_dir / "05_export_geometry.sh", _export_script(run_dir)),
        _write_script(scripts_dir / "06_render_blender.sh", _render_script(config, run_dir)),
    ]
    return scripts


def _check_tools_script(config: dict[str, Any]) -> str:
    required = " ".join(config["real_chain"]["tools"]["required_cli"])
    return f"""#!/usr/bin/env bash
set -euo pipefail
for tool in {required}; do
  command -v "$tool" >/dev/null || {{ echo "missing required tool: $tool" >&2; exit 1; }}
done
test -f "{config['real_chain']['tools']['threestudio_launch']}" || {{ echo "missing threestudio launch.py" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['triposr_run']}" || {{ echo "missing TripoSR run.py" >&2; exit 1; }}
"""


def _splatfacto_script(config: dict[str, Any], run_dir: Path, target: str) -> str:
    data = config["real_chain"]["data"]
    quality = config["real_chain"]["quality"]
    if target == "object_a":
        source = data.get("object_a_video") or data["object_a_images"]
    else:
        source = data.get("background_video") or data["background_images"]
    process_kind = "video" if str(source).lower().endswith((".mp4", ".mov", ".m4v")) else "images"
    processed = run_dir / "processed" / target
    output = run_dir / "nerfstudio" / target
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{processed}" "{output}"
ns-process-data {process_kind} --data "{source}" --output-dir "{processed}"
ns-train splatfacto-big \\
  --data "{processed}" \\
  --output-dir "{output}" \\
  --max-num-iterations {int(quality['splatfacto_iterations'])} \\
  --pipeline.model.cull_alpha_thresh={quality['cull_alpha_thresh']} \\
  --pipeline.model.continue_cull_post_densification=False \\
  --pipeline.model.use_scale_regularization=True \\
  --viewer.quit-on-train-completion True
"""


def _object_b_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["real_chain"]
    output = run_dir / "object_b_threestudio"
    prompt = chain["object_b"]["prompt"].replace('"', '\\"')
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{output}"
cd "{Path(chain['tools']['threestudio_launch']).parent}"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 \\
  system.prompt_processor.prompt="{prompt}" \\
  trainer.max_steps={int(chain['object_b']['max_steps'])} \\
  system.guidance.grad_clip=[0,0.5,2.0,10000] \\
  system.prompt_processor.use_perp_neg=true \\
  trial_dir="{output}"
"""


def _object_c_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["real_chain"]
    output = run_dir / "object_c_triposr"
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{output}"
python "{chain['tools']['triposr_run']}" "{chain['data']['object_c_image']}" \\
  --output-dir "{output}" \\
  --bake-texture \\
  --texture-resolution {int(chain['object_c']['texture_resolution'])}
"""


def _export_script(run_dir: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
for target in object_a background; do
  config_path=$(find "{run_dir}/nerfstudio/$target" -name config.yml | sort | tail -1)
  test -n "$config_path" || {{ echo "missing nerfstudio config for $target" >&2; exit 1; }}
  mkdir -p "{run_dir}/exports/$target/splat" "{run_dir}/exports/$target/mesh" "{run_dir}/eval/$target"
  ns-eval --load-config "$config_path" --output-path "{run_dir}/eval/$target/metrics.json"
  ns-export gaussian-splat --load-config "$config_path" --output-dir "{run_dir}/exports/$target/splat"
  ns-export tsdf --load-config "$config_path" --output-dir "{run_dir}/exports/$target/mesh"
done
"""


def _render_script(config: dict[str, Any], run_dir: Path) -> str:
    renderer = config["real_chain"]["tools"]["blender_renderer"]
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{run_dir}/renders"
blender -b -P "{renderer}" -- "{run_dir}" "{run_dir}/renders"
"""


def _run_scripts(scripts: list[Path], logs_dir: Path) -> None:
    for script in scripts:
        log_path = logs_dir / f"{script.stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(["bash", str(script)], check=True, stdout=log, stderr=subprocess.STDOUT)


def _collect_outputs(run_dir: Path) -> None:
    manifest = {"outputs": _expected_outputs(run_dir)}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".ply", ".obj", ".glb", ".png", ".mp4", ".gif", ".json"}:
            manifest.setdefault("observed", []).append(path.as_posix())
    save_json(run_dir / "real_output_manifest.json", manifest)


def _expected_outputs(run_dir: Path) -> dict[str, str]:
    return {
        "object_a_splat": f"{run_dir}/exports/object_a/splat/*.ply",
        "object_a_mesh": f"{run_dir}/exports/object_a/mesh/*",
        "background_splat": f"{run_dir}/exports/background/splat/*.ply",
        "background_mesh": f"{run_dir}/exports/background/mesh/*",
        "object_b_mesh": f"{run_dir}/object_b_threestudio/**/*",
        "object_c_mesh": f"{run_dir}/object_c_triposr/**/*",
        "render_video": f"{run_dir}/renders/fused_scene.mp4",
    }


def _write_script(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
