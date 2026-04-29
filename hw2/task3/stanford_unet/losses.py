"""Loss functions for HW2 Task3 semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import IGNORE_INDEX


class DiceLoss(nn.Module):
    """Softmax Dice loss with support for ignored pixels."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

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
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of Cross-Entropy and Dice losses."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted CE + Dice objective."""
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(train_config: dict, num_classes: int, ignore_index: int = IGNORE_INDEX) -> nn.Module:
    """Create the configured segmentation loss."""
    name = str(train_config.get("loss", "ce")).lower()
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if name == "ce_dice":
        return CombinedLoss(num_classes=num_classes, ignore_index=ignore_index)
    raise ValueError(f"Unsupported loss: {name}")
