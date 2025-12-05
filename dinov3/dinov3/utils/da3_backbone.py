"""DA3 backbone wrapper for DINOv3 MIM distillation.

This module provides `build_da3_backbone`, which builds a DA3 teacher
backbone from the Depth Anything 3 project and exposes a simple
`[N, 3, H, W] -> [N, P, D]` interface on top of the DinoV2 encoder.

The implementation intentionally keeps the dependency surface small:
- We only depend on `DepthAnything3` from `depth_anything_3.api`.
- We only keep the DinoV2 encoder as the teacher module that is used
  inside the DINOv3 training loop.

By default, weights are loaded from the Hugging Face repo
"depth-anything/da3-base" unless `da3_cfg.ckpt_path` is set, in which
case that value is forwarded to `DepthAnything3.from_pretrained`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class DA3BackboneConfig:
    """Lightweight view over the DA3 config block.

    We keep this minimal on purpose; the full Hydra/Yacs config object
    is still passed through from the training code. This wrapper is only
    used to make type usage inside this module clearer.
    """

    ckpt_path: str = ""


class DA3BackboneTeacher(nn.Module):
    """Wrap the DA3 DinoV2 encoder to expose patch tokens.

    The underlying encoder is a `depth_anything_3.model.dinov2.DinoV2`
    instance. Its forward returns a tuple of

        (feats, aux_feats) where
        - feats is a sequence of (patch_tokens, camera_tokens)
        - patch_tokens has shape [B, S, P, D]

    We collapse the (B, S) dimensions into a single batch dimension and
    return a tensor of shape [N, P, D].
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

        # Depth Anything 3 wraps a DinoVisionTransformer under
        # encoder.pretrained. We use its attributes to infer embed dim.
        vit = getattr(self.encoder, "pretrained", self.encoder)
        self.embed_dim: int = int(vit.embed_dim)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Tensor of shape [N, 3, H, W]. This should be
                normalized with ImageNet mean/std and resized such that
                the DA3 patch grid matches the student grid (the
                training pipeline already ensures this by using
                `global_crops_clean`).

        Returns:
            Tensor of shape [N, P, D], where P is the number of spatial
            tokens and D is the DA3 embed dim.
        """

        if images.dim() != 4:
            raise ValueError(f"DA3BackboneTeacher expected images of shape [N, 3, H, W], got {tuple(images.shape)}")

        N, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"DA3BackboneTeacher expected 3-channel input, got C={C}")

        # DinoV2 encoder expects [B, S, 3, H, W]. We use B=1, S=N.
        x = images.unsqueeze(0)  # [1, N, 3, H, W]

        feats, _ = self.encoder(x)  # feats: sequence[(patch_tokens, cam_tokens)]
        if not feats:
            raise RuntimeError("DA3 encoder returned no features")

        # Take the last feature level (matching DA3 depth head usage).
        patch_tokens, _cam_tokens = feats[-1]
        # patch_tokens: [B, S, P, D] with B=1, S=N

        if patch_tokens.dim() != 4:
            raise RuntimeError(
                f"Expected DA3 patch tokens to be 4D [B, S, P, D], got shape {tuple(patch_tokens.shape)}"
            )

        # Collapse (B, S) into a batch dimension.
        tokens = patch_tokens.flatten(0, 1)  # [N, P, D]
        return tokens


def _to_da3_backbone_cfg(da3_cfg) -> DA3BackboneConfig:
    """Extract a minimal DA3BackboneConfig from a Hydra/Yacs config.

    We intentionally avoid importing Hydra/OmegaConf here; we only
    access attributes that are known to exist on the training config.
    """

    ckpt_path = getattr(da3_cfg, "ckpt_path", "") 
    return DA3BackboneConfig(ckpt_path=ckpt_path)


def build_da3_backbone(da3_cfg) -> Tuple[nn.Module, int]:
    """Build the DA3 teacher backbone.

    Args:
        da3_cfg: The `cfg.da3` subtree from the main training config.
            Expected fields:
            - ckpt_path: (optional) Hugging Face repo id or local path
              understood by `DepthAnything3.from_pretrained`. If empty,
              defaults to "depth-anything/da3-base".

    Returns:
        model: nn.Module mapping `[N, 3, H, W] -> [N, P, D]`.
        embed_dim: D, the DA3 token dimension.
    """

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:  # pragma: no cover - dependency comes from sibling repo
        raise ImportError(
            "Failed to import DepthAnything3. Make sure the Depth-Anything-3 "
            "repository is installed or its `src` directory is on PYTHONPATH."
        ) from exc

    cfg_view = _to_da3_backbone_cfg(da3_cfg)

    # When ckpt_path is empty, fall back to the public HF repo.
    model_id = cfg_view.ckpt_path or "depth-anything/da3-base"

    # Load DA3 from Hugging Face Hub or local path. We rely on
    # PyTorchModelHubMixin caching so repeated calls are cheap.
    da3 = DepthAnything3.from_pretrained(model_id)
    # We only need the DinoV2 encoder as our teacher.
    encoder = da3.model.backbone

    teacher = DA3BackboneTeacher(encoder=encoder)

    # Freeze by default; the caller will move to the desired device.
    teacher.requires_grad_(False)
    teacher.eval()

    return teacher, teacher.embed_dim
