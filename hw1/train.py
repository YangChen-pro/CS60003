"""Train the HW1 MLP classifier."""

from __future__ import annotations

import argparse

from mlp_hw1.config import build_train_config
from mlp_hw1.trainer import train_model


def parse_args() -> argparse.Namespace:
    """Parse a small set of practical CLI arguments."""
    parser = argparse.ArgumentParser(description="训练 EuroSAT 三层 MLP 分类器")
    parser.add_argument("--preset", default="default", choices=["quick", "default", "full"])
    parser.add_argument("--activation", default=None, choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for training."""
    args = parse_args()
    config = build_train_config(args.preset)
    if args.activation is not None:
        config.activation = args.activation
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    config.force_rebuild_cache = args.rebuild_cache
    train_model(config=config)


if __name__ == "__main__":
    main()
