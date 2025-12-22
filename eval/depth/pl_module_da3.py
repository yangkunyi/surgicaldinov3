"""PyTorch Lightning module for Depth-Anything-3 (DA3) depth-head finetuning on SCARED.

This module:

* Loads a Depth-Anything-3 model (local HF-style directory or HF repo id).
* Freezes the DA3 backbone (DINOv2) and trains only the DA3 depth head.
* Reuses the existing SCARED datasets in ``eval/depth`` (expects ``image``, ``depth``, ``valid_mask``).
* Uses the same ``SigLoss`` and metrics used by ``eval/depth/pl_module.py``.

Notes:
DA3 uses a patch size of 14, so for inputs where H/W are not divisible by 14
(e.g. 256), the head produces outputs at floor(H/14)*14. We upsample the
prediction back to (H, W) before computing loss/metrics to avoid changing the
dataset pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from loguru import logger
from torch import Tensor, nn
import torch.nn.functional as F

import lightning as pl
from torchvision.transforms import v2

from .loss import SigLoss
from .metrics import abs_rel, mae, rmse
from depth_anything_3.api import DepthAnything3


class DA3DPTDepthModule(pl.LightningModule):
    """Lightning module wrapping frozen DA3 backbone + trainable DA3 depth head."""

    def __init__(
        self,
        da3_cfg: Any,
        optim_cfg: Any,
        scheduler_cfg: Optional[Any] = None,
        loss_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            {
                "da3_cfg": da3_cfg,
                "optim_cfg": optim_cfg,
                "scheduler_cfg": scheduler_cfg,
                "loss_cfg": loss_cfg,
            }
        )

        self.da3_cfg = da3_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.loss_cfg = loss_cfg

        self.da3 = self._build_da3(da3_cfg)

        # Train only the depth head by default.
        self._freeze_all()
        self._unfreeze_depth_head()
        self._freeze_backbone_explicit()

        # Fixed for DA3-Base training on SCARED.
        self.depth_key: str = "depth"

        # Resize (short side) for both image and depth.
        self._short_side: int = 224
        self._img_resize = v2.Resize(
            size=self._short_side,
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self._depth_resize = v2.Resize(
            size=self._short_side,
            interpolation=v2.InterpolationMode.NEAREST,
            antialias=False,
        )

        warm_up = bool(getattr(loss_cfg, "warm_up", False)) if loss_cfg is not None else False
        warm_iter = int(getattr(loss_cfg, "warm_iter", 100)) if loss_cfg is not None else 100
        self.loss_fn = SigLoss(warm_up=warm_up, warm_iter=warm_iter)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialized DA3DPTDepthModule (trainable_params={trainable:,} / total_params={total:,})"
        )

    # ------------------------------------------------------------------
    # Model loading / freezing
    # ------------------------------------------------------------------



    def _build_da3(self, cfg: Any) -> nn.Module:
        """Load DA3 via its HF mixin.

        cfg must provide:
        - model_dir: local HF-style directory (recommended)
        """


        logger.info(f"Loading DA3-Base from_pretrained(model_dir={cfg.model_id})")
        da3 = DepthAnything3.from_pretrained(cfg.model_id)

        da3.model.train()
        return da3

    def _freeze_all(self) -> None:
        for p in self.da3.parameters():
            p.requires_grad = False

    def _unfreeze_depth_head(self) -> None:
        if not hasattr(self.da3, "model") or not hasattr(self.da3.model, "head"):
            raise AttributeError("DA3 object is missing da3.model.head")
        for p in self.da3.model.head.parameters():
            p.requires_grad = True
        self.da3.model.head.train()

    def _freeze_backbone_explicit(self) -> None:
        if not hasattr(self.da3, "model") or not hasattr(self.da3.model, "backbone"):
            raise AttributeError("DA3 object is missing da3.model.backbone")
        for p in self.da3.model.backbone.parameters():
            p.requires_grad = False
        self.da3.model.backbone.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _resize_batch_images(self, images: Tensor) -> Tensor:
        return self._img_resize(images)

    def _resize_batch_depth(self, depth: Tensor) -> Tensor:
        if depth.dim() != 3:
            raise ValueError(f"Expected depth [B,H,W], got {tuple(depth.shape)}")
        return self._depth_resize(depth.unsqueeze(1)).squeeze(1)

    def _extract_depth(self, out: Mapping[str, Tensor], *, H: int, W: int) -> Tensor:
        if self.depth_key not in out:
            raise KeyError(
                f"DA3 head output missing '{self.depth_key}'. Available keys: {list(out.keys())}"
            )
        depth = out[self.depth_key]

        # Handle various DA3 head output layouts.
        # Common cases:
        # - [B, S, H, W]
        # - [B, S, 1, H, W]
        # - [B, 1, H, W]
        # - [B, H, W, 1]
        if depth.dim() == 5 and depth.shape[2] == 1:
            depth = depth.squeeze(2)
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        # Remove sequence/channel singleton dims.
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)

        if depth.dim() != 3:
            raise ValueError(f"Unexpected depth shape {tuple(depth.shape)}")

        # Upsample to match target resolution if needed.
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)

        # Ensure strictly positive domain for log-based loss.
        return torch.clamp(depth, min=1.0e-4)

    def forward(self, images: Tensor) -> Tensor:
        """image -> depth.

        Args:
            images: [B, 3, H, W]
        Returns:
            depth: [B, H, W]
        """

        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        images = self._resize_batch_images(images)

        B, _, H, W = images.shape
        x = images.unsqueeze(1)  # [B, 1, 3, H, W] (DA3 expects B,S,...)

        # Single forward through the DA3 network (backbone is frozen via requires_grad=False).
        # Keep this simple for the fixed DA3-Base + monocular SCARED training use-case.
        out = self.da3.model(x)

        depth = self._extract_depth(out, H=H, W=W)
        if depth.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: images B={B} but depth B={depth.shape[0]}"
            )
        return depth

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        images: Tensor = batch["image"]
        depth_gt: Tensor = batch["depth"]
        valid_mask: Optional[Tensor] = batch.get("valid_mask")

        depth_gt = self._resize_batch_depth(depth_gt)
        valid_mask = (depth_gt >= 0.0001) & (depth_gt <= 150.0)


        depth_pred = self(images)
        loss = self.loss_fn(depth_pred, depth_gt, valid_mask)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=depth_pred.shape[0],
        )

        train_mae = mae(depth_pred, depth_gt, valid_mask)
        train_rmse = rmse(depth_pred, depth_gt, valid_mask)
        train_abs_rel = abs_rel(depth_pred, depth_gt, valid_mask)

        self.log(
            "train/mae",
            train_mae,
            on_step=True,
            on_epoch=True,
            batch_size=depth_pred.shape[0],
        )
        self.log(
            "train/rmse",
            train_rmse,
            on_step=True,
            on_epoch=True,
            batch_size=depth_pred.shape[0],
        )
        self.log(
            "train/abs_rel",
            train_abs_rel,
            on_step=True,
            on_epoch=True,
            batch_size=depth_pred.shape[0],
        )

        if self.trainer is not None and self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            optim_obj = getattr(opt, "optimizer", opt)
            lr = optim_obj.param_groups[0].get("lr")
            if lr is not None:
                self.log(
                    "train/lr",
                    lr,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=depth_pred.shape[0],
                )

        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        images: Tensor = batch["image"]
        depth_gt: Tensor = batch["depth"]
        valid_mask: Optional[Tensor] = batch.get("valid_mask")

        depth_gt = self._resize_batch_depth(depth_gt)
        valid_mask = (depth_gt >= 0.0001) & (depth_gt <= 150.0)

        depth_pred = self(images)

        # Match the existing validation behavior: median scaling per-image.
        depth_pred_4d = depth_pred.unsqueeze(1)
        depth_gt_4d = depth_gt.unsqueeze(1)
        if valid_mask is None:
            valid_mask_4d = torch.ones_like(depth_gt_4d, dtype=torch.bool)
        else:
            valid_mask_4d = valid_mask.unsqueeze(1)

        B, C, H, W = depth_pred_4d.shape
        pred_flat = depth_pred_4d.reshape(B, -1).clone()
        gt_flat = depth_gt_4d.reshape(B, -1).clone()
        mask_flat = valid_mask_4d.reshape(B, -1)
        pred_flat[~mask_flat] = float("nan")
        gt_flat[~mask_flat] = float("nan")

        pred_median = torch.nanmedian(pred_flat, dim=1).values
        gt_median = torch.nanmedian(gt_flat, dim=1).values

        ratio = gt_median / (pred_median + 1e-8)
        ratio[~torch.isfinite(ratio)] = 1.0
        ratio = ratio.view(B, 1, 1, 1)

        depth_pred_scaled = depth_pred_4d * ratio
        depth_pred_scaled = torch.clamp(depth_pred_scaled, min=0.0001, max=150.0)

        loss = self.loss_fn(depth_pred_scaled, depth_gt_4d, valid_mask_4d)

        val_mae = mae(depth_pred_scaled, depth_gt_4d, valid_mask_4d)
        val_rmse = rmse(depth_pred_scaled, depth_gt_4d, valid_mask_4d)
        val_abs_rel = abs_rel(depth_pred_scaled, depth_gt_4d, valid_mask_4d)

        if torch.isfinite(loss):
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=depth_pred.shape[0],
            )
            self.log(
                "val/mae",
                val_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=depth_pred.shape[0],
            )
            self.log(
                "val/rmse",
                val_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=depth_pred.shape[0],
            )
            self.log(
                "val/abs_rel",
                val_abs_rel,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=depth_pred.shape[0],
            )

        return loss

    def configure_optimizers(self):  # type: ignore[override]
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.optim_cfg.lr,
            betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
            weight_decay=self.optim_cfg.weight_decay,
        )

        if (
            self.scheduler_cfg is None
            or getattr(self.scheduler_cfg, "type", "").lower() != "warmuponecyclelr"
        ):
            return optimizer

        if self.trainer is None:
            raise RuntimeError(
                "Trainer is not attached; cannot compute total steps for scheduler."
            )

        total_steps = int(self.trainer.max_steps)
        warmup_fraction = float(self.scheduler_cfg.warmup_fraction)
        warmup_iters = int(total_steps * warmup_fraction)
        pct_start = float(warmup_iters) / float(total_steps) if total_steps > 0 else 0.0

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.optim_cfg.lr,
            total_steps=total_steps,
            pct_start=pct_start,
            final_div_factor=self.scheduler_cfg.final_div_factor,
            anneal_strategy="cos",
            base_momentum=self.scheduler_cfg.base_momentum,
            max_momentum=self.scheduler_cfg.max_momentum,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "one_cycle",
            },
        }
