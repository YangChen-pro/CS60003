"""AI-asset preview renderer for HW3 Task1.

This preview is intentionally separate from the real 3DGS/SDS chain. It gives
an immediate high-quality fused render from the temporary AI A/C assets while
the final phone-capture and dataset inputs are still unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from task1_3dgs_aigc.utils import copy_source_config, make_run_dir, save_json


def run_preview_chain(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare or run the AI-assets high-quality preview renderer."""
    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    scripts_dir = run_dir / "scripts"
    logs_dir = run_dir / "logs"
    renders_dir = run_dir / "renders"
    for directory in [scripts_dir, logs_dir, renders_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    copy_source_config(config["config_path"], run_dir)
    save_json(run_dir / "config.json", config)

    issues = _validate_inputs(config)
    scripts = _write_scripts(config, run_dir, scripts_dir)
    mode = config["preview_chain"]["execution"]["mode"]
    summary = {
        "status": "RUNNING" if mode == "run" else ("READY" if not issues else "NEEDS_INPUTS"),
        "stage": "ai_assets_high_quality_preview",
        "run_dir": run_dir.as_posix(),
        "script_count": len(scripts),
        "scripts": [path.as_posix() for path in scripts],
        "input_issues": issues,
        "expected_outputs": _expected_outputs(run_dir),
        "preview_policy": "AI assets preview only; final submission still requires real 3DGS/SDS/TripoSR outputs.",
    }
    save_json(run_dir / "summary.json", summary)
    if mode == "run":
        if issues:
            raise ValueError("; ".join(issues))
        for path in _expected_outputs(run_dir).values():
            output_path = Path(path)
            if output_path.exists():
                output_path.unlink()
        _run_scripts(scripts, logs_dir)
        missing = [path for path in _expected_outputs(run_dir).values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Preview render did not create expected outputs: {missing}")
        summary["status"] = "PASS"
        summary["observed_outputs"] = _observed_outputs(run_dir)
        save_json(run_dir / "summary.json", summary)
    return summary


def _validate_inputs(config: dict[str, Any]) -> list[str]:
    data = config["preview_chain"]["data"]
    issues: list[str] = []
    object_a_dir = Path(data["object_a_images"])
    if not object_a_dir.is_dir():
        issues.append(f"Missing object A image directory: {object_a_dir}")
    elif not _image_files(object_a_dir):
        issues.append(f"Object A image directory has no PNG/JPG files: {object_a_dir}")
    object_c_image = Path(data["object_c_image"])
    if not object_c_image.is_file():
        issues.append(f"Missing object C image: {object_c_image}")
    renderer = Path(config["preview_chain"]["render"]["blender_renderer"])
    if not renderer.is_file():
        issues.append(f"Missing Blender preview renderer: {renderer}")
    if shutil.which("blender") is None:
        issues.append("Missing required tool: blender")
    return issues


def _write_scripts(config: dict[str, Any], run_dir: Path, scripts_dir: Path) -> list[Path]:
    return [
        _write_script(scripts_dir / "00_check_preview_tools.sh", _check_tools_script(config)),
        _write_script(scripts_dir / "01_render_ai_assets_preview.sh", _render_script(config, run_dir)),
    ]


def _check_tools_script(config: dict[str, Any]) -> str:
    renderer = config["preview_chain"]["render"]["blender_renderer"]
    return f"""#!/usr/bin/env bash
set -euo pipefail
command -v blender >/dev/null || {{ echo "missing required tool: blender" >&2; exit 1; }}
test -f "{renderer}" || {{ echo "missing preview Blender renderer" >&2; exit 1; }}
"""


def _render_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["preview_chain"]
    render = chain["render"]
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{run_dir}/renders"
if command -v xvfb-run >/dev/null; then
  BLENDER_CMD=(xvfb-run -a blender)
else
  BLENDER_CMD=(blender)
fi
"${{BLENDER_CMD[@]}}" -b --python-exit-code 1 -P "{render['blender_renderer']}" -- \\
  "{run_dir}" \\
  "{run_dir}/renders" \\
  "{chain['data']['object_a_images']}" \\
  "{chain['data']['object_c_image']}" \\
  "{int(render['resolution_x'])}" \\
  "{int(render['resolution_y'])}" \\
  "{int(render['frames'])}" \\
  "{int(render['samples'])}"
"""


def _run_scripts(scripts: list[Path], logs_dir: Path) -> None:
    for script in scripts:
        log_path = logs_dir / f"{script.stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(["bash", str(script)], check=True, stdout=log, stderr=subprocess.STDOUT)


def _expected_outputs(run_dir: Path) -> dict[str, str]:
    return {
        "hero_frame": f"{run_dir}/renders/preview_hero.png",
        "render_video": f"{run_dir}/renders/fused_scene.mp4",
        "metadata": f"{run_dir}/renders/preview_metadata.json",
    }


def _observed_outputs(run_dir: Path) -> list[str]:
    return sorted(
        path.as_posix()
        for path in (run_dir / "renders").glob("*")
        if path.suffix.lower() in {".png", ".mp4", ".json"}
    )


def _write_script(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
