from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear layer (ported from dinov3 segmentation linear_head.py)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        n_output_channels: int,
        use_batchnorm: bool = True,
        use_cls_token: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = int(sum(in_channels))
        if use_cls_token:
            self.channels *= 2
        self.n_output_channels = int(n_output_channels)
        self.use_cls_token = bool(use_cls_token)
        self.batchnorm_layer = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity(self.channels)
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout2d(dropout)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 0.0)

    def _transform_inputs(self, inputs: list[Tensor]) -> Tensor:
        inputs = [
            F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs) -> Tensor:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs) -> Tensor:
        output = self._forward_feature(inputs)
        output = self.dropout(output)
        output = self.batchnorm_layer(output)
        output = self.conv(output)
        return output

    def predict(self, inputs, rescale_to: tuple[int, int] = (512, 512)) -> Tensor:
        x = self._forward_feature(inputs)
        x = self.batchnorm_layer(x)
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear")
        return x


class DINOv3FeatureAdapter(nn.Module):
    """Extracts patch-token features from a DINOv3 backbone as spatial maps."""

    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: Sequence[int],
        use_cls_token: bool,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.use_cls_token = bool(use_cls_token)
        self.norm = bool(norm)

    def forward(self, images: Tensor):
        with torch.inference_mode():
            feats = self.backbone.get_intermediate_layers(
                images,
                n=self.layer_indices,
                reshape=True,
                return_class_token=self.use_cls_token,
                return_extra_tokens=False,
                norm=self.norm,
            )
        return feats
