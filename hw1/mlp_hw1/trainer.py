"""Training, evaluation, and search loops."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .backend import get_array_module, seed_everything, to_numpy
from .config import SearchConfig, TrainConfig
from .data import DataSplit, iterate_minibatches, load_dataset, normalize_images
from .metrics import accuracy_score, confusion_matrix, per_class_accuracy
from .model import ThreeLayerMLP
from .reporting import build_report
from .visualization import (
    plot_confusion_matrix,
    plot_first_layer_weights,
    plot_misclassified_examples,
    plot_training_curves,
)


def train_model(
    config: TrainConfig,
    run_name: str | None = None,
    generate_reports: bool = True,
) -> dict:
    """Train a model and persist artifacts."""
    xp = get_array_module()
    seed_everything(config.seed)

    dataset = load_dataset(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        limit_per_class=config.limit_per_class,
        force_rebuild=config.force_rebuild_cache,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / "runs" / (run_name or f"{timestamp}_cupy_{config.activation}_h{config.hidden_dim}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ThreeLayerMLP(
        input_dim=dataset.train.images.shape[1],
        hidden_dim=config.hidden_dim,
        hidden_dim2=config.resolved_hidden_dim2(),
        output_dim=len(dataset.class_names),
        activation=config.activation,
        xp=xp,
        seed=config.seed,
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_checkpoint = run_dir / "best_model.npz"
    best_val_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate / (1.0 + config.lr_decay * (epoch - 1))
        # 梯度与参数更新都在这里手写完成，不依赖任何现成训练框架。
        for batch_images, batch_labels in iterate_minibatches(
            dataset.train.images,
            dataset.train.labels,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            shuffle=True,
        ):
            features = normalize_images(batch_images, dataset.mean, dataset.std)
            batch_x = xp.asarray(features)
            batch_y = xp.asarray(batch_labels)
            model.loss_and_backward(batch_x, batch_y, config.weight_decay)
            model.step(learning_rate, grad_clip=config.grad_clip)

        train_metrics = evaluate_split(model, dataset.train, dataset.mean, dataset.std, config.eval_batch_size)
        val_metrics = evaluate_split(model, dataset.val, dataset.mean, dataset.std, config.eval_batch_size)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(learning_rate)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            model.save(best_checkpoint)

    best_model = ThreeLayerMLP.load(best_checkpoint, xp)
    test_metrics = evaluate_split(
        best_model,
        dataset.test,
        dataset.mean,
        dataset.std,
        config.eval_batch_size,
        return_predictions=True,
    )
    matrix = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], len(dataset.class_names))
    summary = {
        "backend": "cupy",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "class_names": dataset.class_names,
        "per_class_accuracy": per_class_accuracy(matrix).tolist(),
        "train_samples": int(dataset.train.images.shape[0]),
        "val_samples": int(dataset.val.images.shape[0]),
        "test_samples": int(dataset.test.images.shape[0]),
        "config": config.to_dict(),
    }

    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "confusion_matrix.json").write_text(
        json.dumps(matrix.tolist(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if generate_reports:
        plot_training_curves(history, run_dir / "training_curves.png")
        plot_confusion_matrix(matrix, dataset.class_names, run_dir / "confusion_matrix.png")
        plot_first_layer_weights(to_numpy(best_model.w1), dataset.image_shape, run_dir / "first_layer_weights.png")
        plot_misclassified_examples(
            dataset.test.images,
            test_metrics["y_true"],
            test_metrics["y_pred"],
            dataset.class_names,
            dataset.image_shape,
            run_dir / "misclassified_examples.png",
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(matrix)
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "summary": summary,
        "history": history,
        "confusion_matrix": matrix,
    }


def evaluate_model(
    config: TrainConfig,
    checkpoint_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate a saved model on the test split."""
    xp = get_array_module()
    dataset = load_dataset(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        limit_per_class=config.limit_per_class,
        force_rebuild=config.force_rebuild_cache,
    )
    model = ThreeLayerMLP.load(checkpoint_path, xp)
    metrics = evaluate_split(
        model,
        dataset.test,
        dataset.mean,
        dataset.std,
        config.eval_batch_size,
        return_predictions=True,
    )
    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], len(dataset.class_names))
    destination = output_dir or checkpoint_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "class_names": dataset.class_names,
                "per_class_accuracy": per_class_accuracy(matrix).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_confusion_matrix(matrix, dataset.class_names, destination / "evaluation_confusion_matrix.png")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(matrix)
    return {
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "confusion_matrix": matrix,
    }


def run_search(config: SearchConfig) -> dict:
    """Run a simple grid/random search over hyper-parameters."""
    candidates = [
        {
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "hidden_dim2": hidden_dim2,
            "weight_decay": weight_decay,
            "lr_decay": lr_decay,
            "grad_clip": grad_clip,
            "activation": activation,
        }
        for learning_rate in config.learning_rates
        for hidden_dim in config.hidden_dims
        for hidden_dim2 in config.hidden_dims2
        for weight_decay in config.weight_decays
        for lr_decay in config.lr_decays
        for grad_clip in config.grad_clips
        for activation in config.activations
    ]
    if config.strategy == "random":
        rng = np.random.default_rng(config.train_config.seed)
        rng.shuffle(candidates)
    candidates = candidates[: config.max_trials]

    search_dir = config.train_config.output_dir / "search" / datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "search_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    best_row: dict | None = None
    for trial_id, candidate in enumerate(candidates, start=1):
        trial_config = replace(
            config.train_config,
            learning_rate=candidate["learning_rate"],
            hidden_dim=candidate["hidden_dim"],
            hidden_dim2=candidate["hidden_dim2"],
            weight_decay=candidate["weight_decay"],
            lr_decay=candidate["lr_decay"],
            grad_clip=candidate["grad_clip"],
            activation=candidate["activation"],
        )
        run_result = train_model(
            config=trial_config,
            run_name=f"trial_{trial_id:02d}",
            generate_reports=False,
        )
        row = {
            "trial": trial_id,
            **candidate,
            "best_val_accuracy": run_result["summary"]["best_val_accuracy"],
            "test_accuracy": run_result["summary"]["test_accuracy"],
            "run_dir": run_result["run_dir"],
        }
        rows.append(row)
        if best_row is None or row["best_val_accuracy"] > best_row["best_val_accuracy"]:
            best_row = row
        print(
            f"[Trial {trial_id:02d}] "
            f"lr={row['learning_rate']:.4f} "
            f"hidden=({row['hidden_dim']},{row['hidden_dim2']}) "
            f"decay={row['lr_decay']:.4f} "
            f"wd={row['weight_decay']:.4e} "
            f"clip={row['grad_clip']:.1f} "
            f"act={row['activation']} "
            f"val_acc={row['best_val_accuracy']:.4f}"
        )

    with (search_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (search_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_row is not None:
        (search_dir / "best_result.json").write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
        top_trials = sorted(rows, key=lambda item: item["best_val_accuracy"], reverse=True)[:5]
        best_run_dir = Path(best_row["run_dir"])
        best_summary = json.loads((best_run_dir / "summary.json").read_text(encoding="utf-8"))
        build_report(best_run_dir=best_run_dir, best_summary=best_summary, search_dir=search_dir, top_trials=top_trials)
    return {
        "search_dir": str(search_dir),
        "results": rows,
        "best_result": best_row,
    }


def evaluate_split(
    model: ThreeLayerMLP,
    split: DataSplit,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    return_predictions: bool = False,
) -> dict:
    """Evaluate a split in mini-batches."""
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_images, batch_labels in iterate_minibatches(
        split.images,
        split.labels,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
    ):
        features = normalize_images(batch_images, mean, std)
        batch_x = model.xp.asarray(features)
        batch_y = model.xp.asarray(batch_labels)
        losses.append(model.compute_loss(batch_x, batch_y))
        preds = to_numpy(model.predict(batch_x))
        predictions.append(preds.astype(np.int64))
        targets.append(batch_labels.astype(np.int64))

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    result = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if return_predictions:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
    return result
