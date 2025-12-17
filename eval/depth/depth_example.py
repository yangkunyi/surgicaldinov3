import os
import glob
import numpy as np
import cv2
import tqdm
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def depth_evaluation(gt_depths, pred_depths, savedir=None, pred_masks=None, min_depth=0.0001, max_depth=100):
    assert gt_depths.shape[0] == pred_depths.shape[0]

    gt_depths_valid = []
    pred_depths_valid = []
    errors = []
    num = gt_depths.shape[0]
    for i in range(num):
        gt_depth = gt_depths[i]
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        if pred_masks is not None:
            pred_mask = pred_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_width, gt_height)) > 0.5
            mask = mask * pred_mask

        if mask.sum() == 0:
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)

    ratio = np.median(np.concatenate(gt_depths_valid)) / \
                np.median(np.concatenate(pred_depths_valid))

    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    return mean_errors


def build_lance_dataloader(
    lance_path,
    dataset_ids,
    image_size=256,
    min_depth=0.0001,
    max_depth=150.0,
    batch_size=32,
    num_workers=8,
):
    from torch.utils.data import DataLoader
    from .scared_lance import LanceMapDataset

    ds = LanceMapDataset(
        lance_path,
        allowed_ids=dataset_ids,
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

