"""Configuration objects and preset builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "EuroSAT_RGB"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class TrainConfig:
    """Training configuration."""

    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    seed: int = 42
    hidden_dim: int = 512
    activation: str = "relu"
    batch_size: int = 256
    eval_batch_size: int = 512
    epochs: int = 12
    learning_rate: float = 0.02
    lr_decay: float = 0.05
    weight_decay: float = 1e-4
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    limit_per_class: int | None = None
    force_rebuild_cache: bool = False

    def to_dict(self) -> dict:
        """Serialize the config to a JSON-friendly dict."""
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class SearchConfig:
    """Hyper-parameter search configuration."""

    train_config: TrainConfig
    strategy: str = "grid"
    max_trials: int = 8
    learning_rates: tuple[float, ...] = (0.03, 0.05, 0.08)
    hidden_dims: tuple[int, ...] = (256, 512, 768)
    weight_decays: tuple[float, ...] = (1e-4, 5e-4, 1e-3)
    activations: tuple[str, ...] = ("relu", "tanh")

    def to_dict(self) -> dict:
        """Serialize the config to a JSON-friendly dict."""
        return {
            "train_config": self.train_config.to_dict(),
            "strategy": self.strategy,
            "max_trials": self.max_trials,
            "learning_rates": list(self.learning_rates),
            "hidden_dims": list(self.hidden_dims),
            "weight_decays": list(self.weight_decays),
            "activations": list(self.activations),
        }


def build_train_config(preset: str) -> TrainConfig:
    """Build a preset training config."""
    normalized = preset.lower()
    if normalized == "quick":
        return TrainConfig(
            hidden_dim=128,
            batch_size=128,
            eval_batch_size=256,
            epochs=2,
            learning_rate=0.03,
            lr_decay=0.1,
            weight_decay=1e-4,
            limit_per_class=120,
        )
    if normalized == "default":
        return TrainConfig()
    if normalized == "full":
        return TrainConfig(
            hidden_dim=768,
            batch_size=256,
            eval_batch_size=512,
            epochs=20,
            learning_rate=0.015,
            lr_decay=0.03,
            weight_decay=5e-4,
        )
    raise ValueError(f"未知预设: {preset}")


def build_search_config(preset: str) -> SearchConfig:
    """Build a preset hyper-parameter search config."""
    train_config = build_train_config("quick" if preset == "quick" else "default")
    if preset == "quick":
        train_config.epochs = 2
        train_config.limit_per_class = 100
        return SearchConfig(
            train_config=train_config,
            strategy="grid",
            max_trials=4,
            learning_rates=(0.01, 0.03),
            hidden_dims=(128, 256),
            weight_decays=(1e-4, 5e-4),
            activations=("relu", "tanh"),
        )
    return SearchConfig(train_config=train_config)
