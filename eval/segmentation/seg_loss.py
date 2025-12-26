from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def dice_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int,
    smooth: float,
    exponent: float,
) -> Tensor:
    probs = F.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    one_hot = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2)
    valid = (target != ignore_index).unsqueeze(1)

    probs = probs * valid
    one_hot = one_hot * valid

    probs = probs.flatten(2)
    one_hot = one_hot.flatten(2)

    inter = (probs * one_hot).sum(dim=2)
    den = probs.pow(exponent).sum(dim=2) + one_hot.pow(exponent).sum(dim=2)
    dice = (2.0 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Cross-entropy with optional Dice, following dinov3 segmentation loss.py."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.ignore_index = int(cfg.ignore_index)
        self.ce_weight = float(cfg.ce_weight)
        self.dice_weight = float(cfg.dice_weight)
        self.dice_smooth = float(cfg.dice_smooth)
        self.dice_exponent = float(cfg.dice_exponent)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, target.long(), ignore_index=self.ignore_index)
        loss = self.ce_weight * ce
        if self.dice_weight > 0.0:
            loss = loss + self.dice_weight * dice_loss(
                logits,
                target,
                ignore_index=self.ignore_index,
                smooth=self.dice_smooth,
                exponent=self.dice_exponent,
            )
        return loss

