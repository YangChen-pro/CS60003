#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
CHECKPOINT=${2:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
NAME=${3:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
ENV_PY=/data/yc/miniconda/envs/llm-26-gpu/bin/python
cd /data/yc/CS60003
export PYTHONPATH=/data/yc/CS60003/hw3/task2/src:${PYTHONPATH:-}
$ENV_PY -m hw3_task2.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT" --name "$NAME" --output "hw3/task2/outputs/eval/${NAME}_splitD"
