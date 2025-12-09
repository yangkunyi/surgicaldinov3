from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    diff = torch.abs(pred - target)
    diff = diff[mask]
    return diff.mean()


def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    diff2 = (pred - target) ** 2
    diff2 = diff2[mask]
    return torch.sqrt(diff2.mean() + 1e-12)


def abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    t = target.clone()
    t[t == 0] = 1e-6
    rel = torch.abs(pred - target) / t
    rel = rel[mask]
    return rel.mean()

