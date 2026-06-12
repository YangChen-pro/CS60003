#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json>}
REPO_ID=${2:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json>}
OUTPUT=${3:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json>}
ENV_PY=/data/yc/miniconda/envs/llm-26-gpu/bin/python
cd /data/yc/CS60003
export PYTHONPATH=/data/yc/CS60003/hw3/task2/src:${PYTHONPATH:-}
$ENV_PY -m hw3_task2.upload_modelscope --model-dir "$MODEL_DIR" --repo-id "$REPO_ID" --output "$OUTPUT" --secret-env /data/yc/CS60003/.helloagents/secrets/hw3.env
