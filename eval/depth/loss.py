from enum import Enum
from functools import partial

import torch
from torch import nn

class SigLoss(nn.Module):
    """Sigloss

    Adapted from Binsformer who adapted from AdaBins
    https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/7c0c89c8db07631fec1737f3087e4f1f540ccd53/depth/models/losses/sigloss.py#L8
    """

    def __init__(self, warm_up=True, warm_iter=100):
        super(SigLoss, self).__init__()
        self.loss_name = "SigLoss"
        self.eps = 0.001  # avoid grad explode
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=bool)
        input = input[valid_mask]
        target = target[valid_mask]

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = 0.15 * torch.pow(torch.mean(g), 2)
        if self.warm_up and self.warm_up_counter < self.warm_iter:
            self.warm_up_counter += 1
        else:
            Dg += torch.var(g)
        if Dg <= 0:
            return torch.abs(Dg)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, valid_mask=None):
        """Forward function."""

        return self.sigloss(depth_pred, depth_gt, valid_mask)