# HW2 Task2: Road Vehicle Detection and Tracking

This folder contains the code for HW2 Task2. The detector is a YOLOv8s model
fine-tuned on the Road Vehicle Images dataset. The tracking script runs
YOLOv8 + ByteTrack, draws stable display IDs, and counts vehicles crossing a
virtual line.

## Files

```text
hw2/task2/
├── configs/
├── road_yolo/
├── tests/
├── train.py
├── evaluate.py
├── track_video.py
├── upload_swanlab_history.py
└── outputs/              # local only, ignored by Git
```

## Data

```text
hw2/RoadVehicleImages/trafic_data/
├── train/images/   (2704 images)
├── train/labels/
├── valid/images/   (300 images)
├── valid/labels/
└── data_hw2.yaml
```

The dataset has 21 vehicle classes in YOLO format.

## Environment

Install PyTorch with the CUDA build that matches the machine, then install the
remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r hw2/task2/requirements.txt
```

Quick check:

```bash
python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

## Training

Run from the repository root:

```bash
export SWANLAB_API_KEY=<your_key>
python hw2/task2/train.py \
  --config hw2/task2/configs/yolov8s_baseline.yaml \
  --device auto
```

Each run writes to `hw2/task2/outputs/{timestamp}_{name}/`:

- `source_config.yaml`, `config.json`, `env.json`
- `history.csv`, `curves.png`
- `best.pt`
- `metrics.json` with `best_epoch`, `best_mAP50`, and `best_mAP50_95`

Final model artifacts are published on ModelScope:

```text
https://modelscope.cn/models/youngchen/CS60003/tree/master/hw2/task2/yolov8s_baseline
```

## Evaluation

```bash
python hw2/task2/evaluate.py \
  --checkpoint hw2/task2/outputs/<run_dir>/best.pt \
  --config hw2/task2/configs/yolov8s_baseline.yaml
```

## Tracking and line-crossing count

```bash
python hw2/task2/track_video.py \
  --model hw2/task2/outputs/<run_dir>/best.pt \
  --video <path/to/video.mp4> \
  --output-dir hw2/task2/outputs/tracking_result \
  --line-x 0.5
```

Outputs:

- `tracked.mp4`: annotated video with boxes, class names, display IDs, and count
- `frames.json`: per-frame detections and running count
- `summary.json`: `total_crossed`, `display_ids_crossed`, and tracking settings
- `snapshots/`: sampled frames for occlusion analysis

Tracking defaults:

- `--conf 0.1` keeps low-confidence boxes for ByteTrack association.
- `configs/bytetrack_occlusion.yaml` raises `track_buffer` to 120 frames.
- New raw track IDs can be merged back to a recent display ID when class
  family, predicted position, and box size are compatible.
- Temporarily hidden tracks can be counted by short-term motion prediction.
  Use `--disable-missing-prediction` to turn this off.

## SwanLab

Project: `cs60003-hw2-task2`. Set `SWANLAB_API_KEY` before logging.

Training history can be replayed after a run:

```bash
export SWANLAB_API_KEY=<your_key>
python hw2/task2/upload_swanlab_history.py \
  --all \
  --project cs60003-hw2-task2 \
  --group task2-history-replay
```

Uploaded runs are listed in `SWANLAB_RUNS.md`.
