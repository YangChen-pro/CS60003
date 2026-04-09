"""EuroSAT data loading and preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SHAPE = (64, 64, 3)


@dataclass
class DataSplit:
    """A dataset split stored in CPU memory."""

    images: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    """Train/validation/test splits and normalization stats."""

    train: DataSplit
    val: DataSplit
    test: DataSplit
    mean: np.ndarray
    std: np.ndarray
    class_names: list[str]
    image_shape: tuple[int, int, int]


def load_dataset(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    limit_per_class: int | None = None,
    force_rebuild: bool = False,
) -> DatasetBundle:
    """Load EuroSAT data from cache or rebuild the cache."""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = "full" if limit_per_class is None else f"limit{limit_per_class}"
    cache_path = cache_dir / f"eurosat_seed{seed}_{limit_tag}.npz"
    meta_path = cache_dir / f"eurosat_seed{seed}_{limit_tag}.json"

    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        return _load_cached_bundle(cache_path, meta_path)

    bundle = _build_dataset(data_dir, seed, val_ratio, test_ratio, limit_per_class)
    np.savez(
        cache_path,
        train_images=bundle.train.images,
        train_labels=bundle.train.labels,
        val_images=bundle.val.images,
        val_labels=bundle.val.labels,
        test_images=bundle.test.images,
        test_labels=bundle.test.labels,
        mean=bundle.mean,
        std=bundle.std,
    )
    meta_path.write_text(
        json.dumps(
            {
                "class_names": bundle.class_names,
                "image_shape": list(bundle.image_shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


def normalize_images(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize uint8 image batches to float32 feature vectors."""
    features = images.astype(np.float32) / 255.0
    return (features - mean) / std


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    """Yield mini-batches from NumPy arrays."""
    indices = np.arange(images.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def _load_cached_bundle(cache_path: Path, meta_path: Path) -> DatasetBundle:
    """Load cached dataset arrays."""
    payload = np.load(cache_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetBundle(
        train=DataSplit(payload["train_images"], payload["train_labels"]),
        val=DataSplit(payload["val_images"], payload["val_labels"]),
        test=DataSplit(payload["test_images"], payload["test_labels"]),
        mean=payload["mean"],
        std=payload["std"],
        class_names=list(meta["class_names"]),
        image_shape=tuple(meta["image_shape"]),
    )


def _build_dataset(
    data_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    limit_per_class: int | None,
) -> DatasetBundle:
    """Build the dataset from image folders."""
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]

    image_paths: list[Path] = []
    labels: list[int] = []
    for class_index, class_dir in enumerate(class_dirs):
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if limit_per_class is not None:
            files = files[:limit_per_class]
        image_paths.extend(files)
        labels.extend([class_index] * len(files))

    images = np.empty((len(image_paths), IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]), dtype=np.uint8)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        images[index] = np.asarray(rgb, dtype=np.uint8).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64)

    train_indices, val_indices, test_indices = _stratified_split(labels_array, val_ratio, test_ratio, seed)
    train_images = images[train_indices]
    train_labels = labels_array[train_indices]
    val_images = images[val_indices]
    val_labels = labels_array[val_indices]
    test_images = images[test_indices]
    test_labels = labels_array[test_indices]

    mean, std = _compute_mean_std(train_images)
    return DatasetBundle(
        train=DataSplit(train_images, train_labels),
        val=DataSplit(val_images, val_labels),
        test=DataSplit(test_images, test_labels),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        class_names=class_names,
        image_shape=IMAGE_SHAPE,
    )


def _compute_mean_std(images: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-set normalization statistics in chunks."""
    total = images.shape[0]
    channel_sum = np.zeros(IMAGE_SHAPE[2], dtype=np.float64)
    channel_sum_squares = np.zeros(IMAGE_SHAPE[2], dtype=np.float64)
    pixel_count = total * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    for start in range(0, total, chunk_size):
        batch = images[start : start + chunk_size].astype(np.float32).reshape(-1, *IMAGE_SHAPE) / 255.0
        channel_sum += batch.sum(axis=(0, 1, 2), dtype=np.float64)
        channel_sum_squares += np.square(batch, dtype=np.float64).sum(axis=(0, 1, 2), dtype=np.float64)
    channel_mean = channel_sum / pixel_count
    channel_variance = np.maximum(channel_sum_squares / pixel_count - np.square(channel_mean), 1e-4)
    channel_std = np.sqrt(channel_variance)
    repeat_count = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    mean = np.tile(channel_mean, repeat_count)
    std = np.tile(channel_std, repeat_count)
    return mean, std


def _stratified_split(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices while preserving class balance."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        class_count = len(class_indices)
        test_count = max(1, int(round(class_count * test_ratio)))
        val_count = max(1, int(round(class_count * val_ratio)))
        if test_count + val_count >= class_count:
            val_count = max(1, class_count // 6)
            test_count = max(1, class_count // 6)
        test_parts.append(class_indices[:test_count])
        val_parts.append(class_indices[test_count : test_count + val_count])
        train_parts.append(class_indices[test_count + val_count :])
    return (
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
    )
