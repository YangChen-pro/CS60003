#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dell/yc/CS60003}
ENV_NAME=${ENV_NAME:-qwen14b}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}

source /home/dell/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install -i "$PIP_INDEX_URL" --upgrade pip setuptools wheel
python -m pip install -i "$PIP_INDEX_URL" nerfstudio gsplat

mkdir -p "$PROJECT_DIR/hw3/task1/external"
cd "$PROJECT_DIR/hw3/task1/external"

if [ ! -d threestudio ]; then
  git clone https://github.com/threestudio-project/threestudio.git
fi
cd threestudio
python -m pip install -i "$PIP_INDEX_URL" -r requirements.txt
python -m pip install -i "$PIP_INDEX_URL" -e .

cd "$PROJECT_DIR/hw3/task1/external"
if [ ! -d TripoSR ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR.git
fi
cd TripoSR
python -m pip install -i "$PIP_INDEX_URL" -r requirements.txt

echo "Install COLMAP, FFmpeg and Blender if they are missing:"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y colmap ffmpeg blender"
