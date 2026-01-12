# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes
from .lora import build_lora_linear


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_targets: List[str] | None = None,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        lora_targets = set(lora_targets or [])
        use_fc1 = lora_rank > 0 and ("mlp" in lora_targets or "ffn" in lora_targets or "fc1" in lora_targets)
        use_fc2 = lora_rank > 0 and ("mlp" in lora_targets or "ffn" in lora_targets or "fc2" in lora_targets)
        self.fc1 = build_lora_linear(
            in_features,
            hidden_features,
            bias=bias,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_fc1,
            base_layer=nn.Linear,
            device=device,
        )
        self.act = act_layer()
        self.fc2 = build_lora_linear(
            hidden_features,
            out_features,
            bias=bias,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_fc2,
            base_layer=nn.Linear,
            device=device,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_targets: List[str] | None = None,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        lora_targets = set(lora_targets or [])
        use_w1 = lora_rank > 0 and ("mlp" in lora_targets or "ffn" in lora_targets or "w1" in lora_targets)
        use_w2 = lora_rank > 0 and ("mlp" in lora_targets or "ffn" in lora_targets or "w2" in lora_targets)
        use_w3 = lora_rank > 0 and ("mlp" in lora_targets or "ffn" in lora_targets or "w3" in lora_targets)
        self.w1 = build_lora_linear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_w1,
            base_layer=nn.Linear,
            device=device,
        )
        self.w2 = build_lora_linear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_w2,
            base_layer=nn.Linear,
            device=device,
        )
        self.w3 = build_lora_linear(
            swiglu_hidden_features,
            out_features,
            bias=bias,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_w3,
            base_layer=nn.Linear,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
