#!/usr/bin/env bash
set -euo pipefail
ENV_PY=/data/yc/miniconda/envs/llm-26-gpu/bin/python
cd /data/yc/CS60003
export PYTHONPATH=/data/yc/CS60003/hw3/task2/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
$ENV_PY -m hw3_task2.train --config hw3/task2/configs/dry_run.yaml --dry-run
