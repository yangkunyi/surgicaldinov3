# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Type

from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        *,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Linear(
            self.in_features,
            r,
            bias=False,
            dtype=base_layer.weight.dtype,
            device=base_layer.weight.device,
        )
        self.lora_B = nn.Linear(
            r,
            self.out_features,
            bias=False,
            dtype=base_layer.weight.dtype,
            device=base_layer.weight.device,
        )
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight) # 关键！必须把 B 的权重设为 0

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        lora_x = self.lora_dropout(x)
        lora_out = self.lora_B(self.lora_A(lora_x))
        return result + (lora_out.to(dtype=result.dtype) * self.scaling)


def build_lora_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    use_lora: bool,
    base_layer: Type[nn.Module] = nn.Linear,
    device=None,
) -> nn.Module:
    base = base_layer(in_features, out_features, bias=bias, device=device)
    if use_lora:
        return LoRALinear(base, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    return base
