from __future__ import annotations

import torch
from torch import Tensor


def calculate_intersect_and_union(
    pred_label: Tensor,
    label: Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pred_label = pred_label.float()
    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]

    area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred_label = torch.histc(pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def compute_iou_and_acc(
    area_intersect: Tensor,
    area_union: Tensor,
    area_label: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    iou = area_intersect / area_union
    acc = area_intersect / area_label
    aacc = area_intersect.sum() / area_label.sum()
    return iou, acc, aacc

