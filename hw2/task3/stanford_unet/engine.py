"""Training and evaluation loops for HW2 Task3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .metrics import summarize_confusion, update_confusion_matrix
from .utils import append_history, plot_history, save_json
from .visualization import save_prediction_grid


@dataclass
class EpochResult:
    """Aggregated segmentation metrics for one epoch."""

    loss: float
    pixel_acc: float
    miou: float
    per_class_iou: dict[str, float | None]
    confusion_matrix: list[list[int]]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    log_interval: int,
    grad_clip_norm: float,
    epoch: int,
    num_classes: int,
) -> EpochResult:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for step, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is None:
            loss.backward()
            _clip_gradients(model, grad_clip_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                _clip_gradients(model, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        confusion = update_confusion_matrix(confusion, logits, masks, num_classes=num_classes)

        if log_interval > 0 and step % log_interval == 0:
            metrics = summarize_confusion(confusion, CLASS_NAMES[:num_classes])
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / total_seen:.4f} miou={metrics['miou']:.4f}",
                flush=True,
            )

    return _epoch_result(total_loss, total_seen, confusion, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    sample_path: Path | None = None,
    max_samples: int = 0,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> EpochResult:
    """Evaluate a segmentation model and optionally save prediction samples."""
    model.eval()
    total_loss = 0.0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    saved_samples = False

    for images, masks, ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        confusion = update_confusion_matrix(confusion, preds, masks, num_classes=num_classes)

        if sample_path is not None and max_samples > 0 and not saved_samples:
            save_prediction_grid(
                images.detach().cpu(),
                masks.detach().cpu(),
                preds.detach().cpu(),
                sample_path,
                mean or [0.485, 0.456, 0.406],
                std or [0.229, 0.224, 0.225],
                list(ids),
                max_samples=max_samples,
            )
            saved_samples = True

    return _epoch_result(total_loss, total_seen, confusion, num_classes)


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device,
    config: dict[str, Any],
    run_dir: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Train a model, save best checkpoint and write run artifacts."""
    train_config = config["train"]
    data_config = config["data"]
    epochs = int(train_config["epochs"])
    num_classes = int(config["model"].get("num_classes", NUM_CLASSES))
    log_interval = int(train_config.get("log_interval", 20))
    grad_clip_norm = float(train_config.get("grad_clip_norm", 0.0))
    patience = int(train_config.get("early_stopping_patience", 0))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    history_csv = run_dir / "history.csv"

    best_miou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            scaler,
            log_interval,
            grad_clip_norm,
            epoch,
            num_classes,
        )
        val_result = evaluate(model, loaders["val"], criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()

        append_history(history_csv, _history_row(epoch, train_result, val_result, optimizer))
        _log_epoch(logger, epoch, train_result, val_result, optimizer)
        _print_epoch_summary(epoch, epochs, train_result, val_result)

        if val_result.miou > best_miou:
            best_miou = val_result.miou
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(best_path, epoch, model, config, val_result)
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early_stopping epoch={epoch} patience={patience}", flush=True)
                break

    curves_path = run_dir / "curves.png"
    plot_history(history_csv, curves_path)
    if logger is not None:
        logger.log_image("report/curves_with_axis_labels", curves_path, "Task3 loss, mIoU and pixel accuracy curves")

    summary = {
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_checkpoint": str(best_path),
    }
    save_json(run_dir / "metrics.json", summary)
    return summary


def _epoch_result(total_loss: float, total_seen: int, confusion: torch.Tensor, num_classes: int) -> EpochResult:
    metrics = summarize_confusion(confusion, CLASS_NAMES[:num_classes])
    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        pixel_acc=float(metrics["pixel_acc"]),
        miou=float(metrics["miou"]),
        per_class_iou=metrics["per_class_iou"],
        confusion_matrix=metrics["confusion_matrix"],
    )


def _history_row(epoch: int, train_result: EpochResult, val_result: EpochResult, optimizer: Optimizer) -> dict[str, str | int]:
    return {
        "epoch": epoch,
        "train_loss": f"{train_result.loss:.6f}",
        "train_miou": f"{train_result.miou:.6f}",
        "train_pixel_acc": f"{train_result.pixel_acc:.6f}",
        "val_loss": f"{val_result.loss:.6f}",
        "val_miou": f"{val_result.miou:.6f}",
        "val_pixel_acc": f"{val_result.pixel_acc:.6f}",
        "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
    }


def _log_epoch(logger: Any | None, epoch: int, train_result: EpochResult, val_result: EpochResult, optimizer: Optimizer) -> None:
    if logger is None:
        return
    payload = {
        "train/loss": train_result.loss,
        "train/miou": train_result.miou,
        "train/pixel_accuracy": train_result.pixel_acc,
        "val/loss": val_result.loss,
        "val/miou": val_result.miou,
        "val/pixel_accuracy": val_result.pixel_acc,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    for class_name, value in val_result.per_class_iou.items():
        if value is not None:
            payload[f"val_iou/{class_name}"] = float(value)
    logger.log(payload, step=epoch)


def _print_epoch_summary(epoch: int, epochs: int, train_result: EpochResult, val_result: EpochResult) -> None:
    print(
        f"epoch={epoch}/{epochs} train_loss={train_result.loss:.4f} "
        f"train_miou={train_result.miou:.4f} val_loss={val_result.loss:.4f} "
        f"val_miou={val_result.miou:.4f} val_pixel_acc={val_result.pixel_acc:.4f}",
        flush=True,
    )


def _save_checkpoint(path: Path, epoch: int, model: nn.Module, config: dict[str, Any], val_result: EpochResult) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config,
            "val": {
                "loss": val_result.loss,
                "miou": val_result.miou,
                "pixel_acc": val_result.pixel_acc,
                "per_class_iou": val_result.per_class_iou,
                "confusion_matrix": val_result.confusion_matrix,
            },
        },
        path,
    )


def _clip_gradients(model: nn.Module, grad_clip_norm: float) -> None:
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
