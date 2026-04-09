"""Configuration objects and preset builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "EuroSAT_RGB"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = PROJECT_ROOT / "REPORT.md"


@dataclass
class TrainConfig:
    """Training configuration."""

    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    seed: int = 42
    hidden_dim: int = 512
    hidden_dim2: int | None = None
    activation: str = "relu"
    batch_size: int = 256
    eval_batch_size: int = 512
    epochs: int = 12
    learning_rate: float = 0.02
    lr_decay: float = 0.05
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    dropout_rate: float = 0.0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    limit_per_class: int | None = None
    force_rebuild_cache: bool = False

    def resolved_hidden_dim2(self) -> int:
        """Return the second hidden width."""
        return self.hidden_dim if self.hidden_dim2 is None else self.hidden_dim2

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
    learning_rates: tuple[float, ...] = (0.008, 0.012, 0.02)
    hidden_dims: tuple[int, ...] = (512, 768, 1024)
    hidden_dims2: tuple[int, ...] = (256, 512, 768)
    weight_decays: tuple[float, ...] = (5e-5, 1e-4, 5e-4)
    lr_decays: tuple[float, ...] = (0.01, 0.03, 0.05)
    grad_clips: tuple[float, ...] = (3.0, 5.0)
    activations: tuple[str, ...] = ("relu", "tanh")

    def to_dict(self) -> dict:
        """Serialize the config to a JSON-friendly dict."""
        return {
            "train_config": self.train_config.to_dict(),
            "strategy": self.strategy,
            "max_trials": self.max_trials,
            "learning_rates": list(self.learning_rates),
            "hidden_dims": list(self.hidden_dims),
            "hidden_dims2": list(self.hidden_dims2),
            "weight_decays": list(self.weight_decays),
            "lr_decays": list(self.lr_decays),
            "grad_clips": list(self.grad_clips),
            "activations": list(self.activations),
        }


def build_train_config(preset: str) -> TrainConfig:
    """Build a preset training config."""
    normalized = preset.lower()
    if normalized == "quick":
        return TrainConfig(
            hidden_dim=128,
            hidden_dim2=128,
            batch_size=128,
            eval_batch_size=256,
            epochs=2,
            learning_rate=0.03,
            lr_decay=0.1,
            weight_decay=1e-4,
            grad_clip=5.0,
            dropout_rate=0.0,
            limit_per_class=120,
        )
    if normalized == "default":
        return TrainConfig(hidden_dim=768, hidden_dim2=512, epochs=18, learning_rate=0.012, lr_decay=0.03)
    if normalized == "full":
        return TrainConfig(
            hidden_dim=1024,
            hidden_dim2=768,
            batch_size=256,
            eval_batch_size=512,
            epochs=28,
            learning_rate=0.01,
            lr_decay=0.02,
            weight_decay=1e-4,
            grad_clip=3.0,
            dropout_rate=0.0,
        )
    if normalized == "best":
        return TrainConfig(
            hidden_dim=1280,
            hidden_dim2=768,
            batch_size=256,
            eval_batch_size=512,
            epochs=44,
            learning_rate=0.012,
            lr_decay=0.01,
            weight_decay=2e-4,
            grad_clip=3.0,
            dropout_rate=0.15,
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
            hidden_dims2=(64, 128),
            weight_decays=(1e-4, 5e-4),
            lr_decays=(0.05, 0.1),
            grad_clips=(3.0, 5.0),
            activations=("relu", "tanh"),
        )
    if preset == "full":
        train_config = build_train_config("full")
        return SearchConfig(
            train_config=train_config,
            strategy="grid",
            max_trials=24,
            learning_rates=(0.006, 0.008, 0.01, 0.012),
            hidden_dims=(768, 1024, 1280),
            hidden_dims2=(384, 512, 768),
            weight_decays=(5e-5, 1e-4, 2e-4),
            lr_decays=(0.01, 0.02, 0.03),
            grad_clips=(2.5, 3.0, 4.0),
            activations=("relu", "tanh"),
        )
    return SearchConfig(train_config=train_config, max_trials=18)
