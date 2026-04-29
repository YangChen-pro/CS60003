"""Loss functions for HW2 Task3 semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import IGNORE_INDEX


class DiceLoss(nn.Module):
    """Softmax Dice loss with support for ignored pixels."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean Dice loss over classes after masking ignored pixels."""
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0).long()
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if self.class_weights is None:
            return 1.0 - dice.mean()
        weights = self.class_weights.to(dice.device)
        weights = weights / weights.mean().clamp_min(1.0e-6)
        return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(1.0e-6)


class CombinedLoss(nn.Module):
    """Weighted combination of Cross-Entropy and Dice losses."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted CE + Dice objective."""
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(train_config: dict, num_classes: int, ignore_index: int = IGNORE_INDEX) -> nn.Module:
    """Create the configured segmentation loss."""
    name = str(train_config.get("loss", "ce")).lower()
    class_weights = _class_weights_tensor(train_config.get("class_weights_values"), num_classes)
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
    if name == "ce_dice":
        return CombinedLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
    raise ValueError(f"Unsupported loss: {name}")


def _class_weights_tensor(values: object, num_classes: int) -> torch.Tensor | None:
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != num_classes:
        raise ValueError("class_weights_values must be a list with one value per class.")
    return torch.tensor([float(value) for value in values], dtype=torch.float32)
