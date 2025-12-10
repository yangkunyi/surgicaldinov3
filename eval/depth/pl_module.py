"""PyTorch Lightning module for DINOv3 + DPT depth estimation on SCARED.

This module:

* Loads a DINOv3 backbone from the local ``dinov3`` repo via ``torch.hub``
  using the same style as ``eval/depth/dinov3_example.py``.
* Optionally loads a fine-tuned teacher checkpoint and strips the
  ``"backbone."`` prefix from its keys.
* Freezes the backbone and trains only the DPT depth head.
* Uses the ``SigLoss`` defined in ``eval/depth/loss.py``.

The Lightning entry script is defined in ``train_scared_depth.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from collections import OrderedDict

import torch
from loguru import logger
from torch import Tensor, nn
import torch.nn.functional as F

import lightning as pl

from .dpt import DPT
from .loss import SigLoss
from .metrics import mae, rmse, abs_rel


class DinoDPTDepthModule(pl.LightningModule):
    """Lightning module wrapping frozen DINOv3 backbone + trainable DPT head.

    During training only the DPT head (and optional post-processing layers)
    receive gradients; the DINOv3 backbone is fully frozen and used under
    ``torch.no_grad()``.
    """

    def __init__(
        self,
        backbone_cfg: Any,
        head_cfg: Any,
        optim_cfg: Any,
        scheduler_cfg: Optional[Any] = None,
        loss_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # Store configs in hparams for logging/repro (avoid storing large strings)
        self.save_hyperparameters(
            {
                "backbone_cfg": backbone_cfg,
                "head_cfg": head_cfg,
                "optim_cfg": optim_cfg,
                "scheduler_cfg": scheduler_cfg,
                "loss_cfg": loss_cfg,
            }
        )

        self.backbone_cfg = backbone_cfg
        self.head_cfg = head_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.loss_cfg = loss_cfg

        logger.info(
            f"Initialized DinoDPTDepthModule with backbone={type(self.backbone_cfg).__name__}"
        )

        # --- build modules ---
        self.backbone = self._build_backbone(backbone_cfg)
        self._freeze_backbone()

        # Ensure DPT input dimension matches DINOv3 embed dim
        self.head = DPT(
            dim_in=head_cfg.dim_in,
            patch_size=head_cfg.patch_size,
            output_dim=head_cfg.output_dim,
            activation=head_cfg.activation,
            conf_activation=head_cfg.conf_activation,
            features=head_cfg.features,
            out_channels=head_cfg.out_channels,
            pos_embed=head_cfg.pos_embed,
            down_ratio=head_cfg.down_ratio,
            head_name=head_cfg.head_name,
            use_sky_head=head_cfg.use_sky_head,
            sky_name=head_cfg.sky_name,
            sky_activation=head_cfg.sky_activation,
            use_ln_for_heads=head_cfg.use_ln_for_heads,
            norm_type=head_cfg.norm_type,
            fusion_block_inplace=head_cfg.fusion_block_inplace,
        )

        self.loss_fn = SigLoss(
            warm_up=self.loss_cfg.warm_up, warm_iter=self.loss_cfg.warm_iter
        )

        # Cache layer indices and patch offset for DPT
        self._layer_indices: Sequence[int] = tuple(backbone_cfg.layer_indices)
        self._patch_start_idx: int = backbone_cfg.patch_start_idx

    # ------------------------------------------------------------------
    # Backbone construction / freezing
    # ------------------------------------------------------------------
    def _build_backbone(self, cfg: Any) -> nn.Module:
        """Load DINOv3 backbone via torch.hub, optionally with teacher weights.

        The logic mirrors ``eval/depth/dinov3_example.py`` so that we can load
        checkpoint files produced by the upstream DINOv3 training scripts.
        """

        import torch.hub  # Imported lazily to avoid side effects at module import

        if not cfg.pretrained_weights_path:
            raise ValueError("pretrained_weights_path must be set in BackboneConfig")

        logger.info(
            f"Loading DINOv3 backbone from repo_dir={cfg.repo_dir} hub_name={cfg.hub_name}"
        )
        backbone = torch.hub.load(
            cfg.repo_dir,
            cfg.hub_name,
            source="local",
            weights=cfg.pretrained_weights_path,
        )

        if cfg.trained_checkpoint:
            logger.info(f"Loading trained checkpoint from {cfg.trained_checkpoint}")
            state = torch.load(cfg.trained_checkpoint, map_location="cpu")
            teacher: Mapping[str, Tensor] | None = state.get("teacher")
            if teacher is None:
                raise KeyError("trained_checkpoint missing 'teacher' key")

            prefix = "backbone."
            new_state: "OrderedDict[str, Tensor]" = OrderedDict(
                (k[len(prefix) :] if k.startswith(prefix) else k, v)
                for k, v in teacher.items()
            )
            missing, unexpected = backbone.load_state_dict(new_state, strict=False)
            if missing:
                logger.warning(
                    f"[DINOv3] Missing keys when loading teacher checkpoint: {list(missing)[:5]}"
                )
            if unexpected:
                logger.warning(
                    f"[DINOv3] Unexpected keys when loading teacher checkpoint: {list(unexpected)[:5]}"
                )

        return backbone

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters and put it in eval mode."""

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    # ------------------------------------------------------------------
    # Core forward: image -> depth prediction
    # ------------------------------------------------------------------
    def forward(self, images: Tensor) -> Tensor:
        """Run a forward pass and return depth predictions.

        Args:
            images: Tensor of shape [B, 3, H, W].

        Returns:
            depth_pred: Tensor of shape [B, H_out, W_out].
        """

        with torch.no_grad():
            feats_tokens: Sequence[Tensor] = self.backbone.get_intermediate_layers(
                images,
                n=self._layer_indices,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )

        # Adapt DINOv3 outputs (B, N, C) into the shape expected by DPT:
        # a list of 4 entries, each entry being a sequence whose first element
        # has shape [B, S, N, C]. For images we treat S=1.
        dpt_feats: List[List[Tensor]] = []
        for feat in feats_tokens:
            if feat.dim() != 3:
                raise ValueError(
                    f"Expected DINOv3 features of shape [B, N, C], got {tuple(feat.shape)}"
                )
            B, N, C = feat.shape
            feat_bs1 = feat.unsqueeze(1)  # [B, 1, N, C]
            dpt_feats.append([feat_bs1])

        H, W = images.shape[-2:]

        head_out: Mapping[str, Tensor] = self.head(
            dpt_feats,
            H=H,
            W=W,
            patch_start_idx=self._patch_start_idx,
            chunk_size=None,
        )

        depth = head_out.get("depth")
        if depth is None:
            raise KeyError("DPT head output is missing 'depth' key")

        # depth: [B, S, H_out, W_out] with S=1 -> squeeze sequence dim
        if depth.dim() != 4:
            raise ValueError(
                f"Unexpected depth output shape {tuple(depth.shape)} (expected [B, S, H, W])"
            )
        depth = depth[:, 0]  # [B, H_out, W_out]
        return depth

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        images: Tensor = batch["image"]
        depth_gt: Tensor = batch["depth"]
        valid_mask: Optional[Tensor] = batch.get("valid_mask")

        depth_pred = self(images)
        loss = self.loss_fn(depth_pred, depth_gt, valid_mask)

        # Loss: log every step for progress bar
        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=depth_pred.shape[0],
        )

        # Metrics: aggregate over epoch to avoid noisy per-step logs
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

        # Log current learning rate for monitoring (first optimizer, first param group).
        if self.trainer is not None and self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            # Optimizer may be wrapped (e.g., LightningOptimizer); get the underlying one if needed.
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

        depth_pred = self(images)
        loss = self.loss_fn(depth_pred, depth_gt, valid_mask)

        # Validation metrics aggregated over the full validation run
        val_mae = mae(depth_pred, depth_gt, valid_mask)
        val_rmse = rmse(depth_pred, depth_gt, valid_mask)
        val_abs_rel = abs_rel(depth_pred, depth_gt, valid_mask)

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
            or self.scheduler_cfg.type.lower() != "warmuponecyclelr"
        ):
            return optimizer
        # Scheduler params are taken from the YAML ``scheduler`` section.
        total_steps = self.scheduler_cfg.total_iter
        warmup_iters = self.scheduler_cfg.warmup_iters
        pct_start = float(warmup_iters) / float(total_steps)

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
