"""Run HW3 Task1 pipeline stages.

The current implementation supports the `smoke_assets` stage. It validates the
AI-generated object A/C placeholders and writes evidence files under outputs/.
The entrypoint keeps the same config-driven style as HW1/HW2 train scripts so
later 3DGS/SDS stages can be added without changing the command surface.
"""

from __future__ import annotations

import argparse
import json

from task1_3dgs_aigc.config import load_config, resolve_paths
from task1_3dgs_aigc.formal_chain import run_formal_chain
from task1_3dgs_aigc.real_chain import run_real_chain
from task1_3dgs_aigc.smoke import run_smoke_test
from task1_3dgs_aigc.swanlab_utils import create_swanlab_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    return parser.parse_args()


def main() -> None:
    """Run one configured Task1 stage."""
    args = parse_args()
    config = load_config(args.config)
    resolve_paths(config, args.output_root)
    stage = str(config["task1"]["stage"])
    if stage == "smoke_assets":
        summary = run_smoke_test(config)
    elif stage == "formal_ai_chain":
        summary = run_formal_chain(config)
    elif stage == "real_high_quality":
        summary = run_real_chain(config)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    _log_summary(config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _log_summary(config: dict, summary: dict) -> None:
    logger = create_swanlab_logger(config, summary["run_dir"])
    try:
        metrics = {"task1/status_pass": 1 if summary["status"] == "PASS" else 0}
        if summary.get("stage") == "real_high_quality":
            metrics.update(
                {
                    "task1/script_count": int(summary["script_count"]),
                    "task1/ready": 1 if summary["status"] in {"READY", "PASS"} else 0,
                }
            )
        elif summary.get("stage") == "formal_ai_chain":
            metrics.update(
                {
                    "task1/asset_count": int(summary["asset_count"]),
                    "task1/fused_point_count": int(summary["fused_point_count"]),
                    "task1/elapsed_seconds": float(summary["elapsed_seconds"]),
                }
            )
        else:
            metrics.update(
                {
                    "task1/image_count": int(summary["image_count"]),
                    "task1/object_a_count": int(summary["object_a_count"]),
                    "task1/object_c_count": int(summary["object_c_count"]),
                    "task1/min_width": int(summary["min_width"]),
                    "task1/min_height": int(summary["min_height"]),
                    "task1/mean_neighbor_rms_diff_64": float(summary["mean_neighbor_rms_diff_64"]),
                }
            )
        logger.log(metrics)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
