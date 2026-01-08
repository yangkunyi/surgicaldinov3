from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from colormap import (
    CLASS_NAME_MAPPING,
    COLOR_CLASS_MAPPING,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from seg_head import (
    DINOv3FeatureAdapter,
    LinearHead,
    PixioFeatureAdapter,
    SAM2FeatureAdapter,
)
from seg_loss import SegmentationLoss
from seg_metrics import calculate_intersect_and_union, compute_iou_and_acc


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[2] / path).resolve())


class LinearProbeSegModule(pl.LightningModule):
    def __init__(
        self,
        backbone_cfg: Any,
        head_cfg: Any,
        loss_cfg: Any,
        optimizer_cfg: Any,
        scheduler_cfg: Any,
        *,
        num_classes: int,
        wandb_log_masks: bool,
        wandb_log_interval: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            {
                "backbone_cfg": backbone_cfg,
                "head_cfg": head_cfg,
                "loss_cfg": loss_cfg,
                "optimizer_cfg": optimizer_cfg,
                "scheduler_cfg": scheduler_cfg,
                "num_classes": num_classes,
                "wandb_log_masks": wandb_log_masks,
                "wandb_log_interval": wandb_log_interval,
            }
        )

        self.backbone_cfg = backbone_cfg
        self.head_cfg = head_cfg
        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.num_classes = int(num_classes)
        self.wandb_log_masks = bool(wandb_log_masks)
        self.wandb_log_interval = int(wandb_log_interval)
        self.freeze_backbone = bool(backbone_cfg.freeze_backbone)

        self.backbone = self._build_backbone(backbone_cfg)
        if self.freeze_backbone:
            self._freeze_backbone()

        embed_dim = int(self.backbone.embed_dim)
        in_channels = [embed_dim for _ in backbone_cfg.layer_indices]
        if backbone_cfg.type.lower() == "dinov3":
            self.feature_adapter = DINOv3FeatureAdapter(
                backbone=self.backbone,
                layer_indices=backbone_cfg.layer_indices,
                use_cls_token=head_cfg.use_cls_token,
                freeze_backbone=self.freeze_backbone,
                norm=backbone_cfg.norm,
            )
        elif backbone_cfg.type.lower() == "pixio":
            self.feature_adapter = PixioFeatureAdapter(
                backbone=self.backbone,
                layer_indices=backbone_cfg.layer_indices,
                use_cls_token=head_cfg.use_cls_token,
                freeze_backbone=self.freeze_backbone,
                norm=backbone_cfg.norm,
            )
        elif backbone_cfg.type.lower() == "sam2":
            self.feature_adapter = SAM2FeatureAdapter(
                backbone=self.backbone,
                layer_indices=backbone_cfg.layer_indices,
                freeze_backbone=self.freeze_backbone,
            )
        else:
            raise ValueError(f"Unsupported backbone.type: {backbone_cfg.type}")
        self.head = LinearHead(
            in_channels=in_channels,
            n_output_channels=self.num_classes,
            use_batchnorm=head_cfg.use_batchnorm,
            use_cls_token=head_cfg.use_cls_token,
            dropout=head_cfg.dropout,
        )

        self.loss_fn = SegmentationLoss(loss_cfg)

        self.ignore_index = int(loss_cfg.ignore_index)

        self.register_buffer(
            "val_area_intersect",
            torch.zeros(self.num_classes, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "val_area_union",
            torch.zeros(self.num_classes, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "val_area_pred",
            torch.zeros(self.num_classes, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "val_area_label",
            torch.zeros(self.num_classes, dtype=torch.float64),
            persistent=False,
        )

    def _build_backbone(self, cfg: Any) -> nn.Module:
        if cfg.type.lower() == "dinov3":
            return self._build_dinov3_backbone(cfg)
        if cfg.type.lower() == "pixio":
            return self._build_pixio_backbone(cfg)
        if cfg.type.lower() == "sam2":
            return self._build_sam2_backbone(cfg)
        raise ValueError(f"Unsupported backbone.type: {cfg.type}")

    def _build_dinov3_backbone(self, cfg: Any) -> nn.Module:
        import torch.hub

        repo_dir = _resolve_path(cfg.repo_dir)
        weights_path = _resolve_path(cfg.pretrained_weights_path)
        backbone = torch.hub.load(
            repo_dir,
            cfg.hub_name,
            source="local",
            weights=weights_path,
        )

        if cfg.trained_checkpoint:
            ckpt_path = _resolve_path(cfg.trained_checkpoint)
            state = torch.load(ckpt_path, map_location="cpu")
            teacher: Mapping[str, Tensor] = state["teacher"]
            prefix = "backbone."
            new_state: "OrderedDict[str, Tensor]" = OrderedDict(
                (k[len(prefix) :] if k.startswith(prefix) else k, v)
                for k, v in teacher.items()
            )
            backbone.load_state_dict(new_state, strict=False)

        return backbone

    def _build_pixio_backbone(self, cfg: Any) -> nn.Module:
        import importlib.util
        import sys

        repo_dir = _resolve_path(cfg.repo_dir)
        sys.path.insert(0, repo_dir)

        try:
            module_path = str(Path(repo_dir) / "pixio.py")
            spec = importlib.util.spec_from_file_location("pixio_models", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        model_fn = getattr(module, cfg.model_name)
        pretrained = None
        if cfg.pretrained_weights_path:
            pretrained = _resolve_path(cfg.pretrained_weights_path)
        backbone = model_fn(pretrained=pretrained)
        if cfg.trained_checkpoint:
            ckpt_path = _resolve_path(cfg.trained_checkpoint)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = state["model"]
            incompatible = backbone.load_state_dict(model_state, strict=False)
            print(
                f"Pixio trained_checkpoint mismatch keys: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        backbone.embed_dim = int(backbone.patch_embed.proj.out_channels)
        return backbone

    def _build_sam2_backbone(self, cfg: Any) -> nn.Module:
        import sys

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        repo_dir = _resolve_path(cfg.repo_dir)
        sys.path.insert(0, repo_dir)
        try:
            config_path = _resolve_path(cfg.config_path)
            sam2_cfg = OmegaConf.load(config_path)
            OmegaConf.resolve(sam2_cfg)
            image_encoder_cfg = sam2_cfg.model.image_encoder
            backbone = instantiate(image_encoder_cfg, _recursive_=True)
        finally:
            sys.path.pop(0)

        weights_path = _resolve_path(cfg.pretrained_weights_path)
        state = torch.load(weights_path, map_location="cpu")
        model_state = state["model"]
        prefix = "image_encoder."
        new_state: "OrderedDict[str, Tensor]" = OrderedDict(
            (k[len(prefix) :], v) for k, v in model_state.items() if k.startswith(prefix)
        )
        backbone.load_state_dict(new_state)
        backbone.embed_dim = int(backbone.neck.d_model)
        return backbone

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:  # type: ignore[override]
        feats = self.feature_adapter(images)
        logits = self.head(feats)
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        images: Tensor = batch["image"]
        masks: Tensor = batch["mask"]
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        preds = logits.argmax(dim=1)
        self._maybe_log_wandb_masks(
            split="train",
            images=images,
            masks=masks,
            preds=preds,
            batch=batch,
            batch_idx=batch_idx,
        )
        return loss

    def on_validation_epoch_start(self) -> None:  # type: ignore[override]
        self.val_area_intersect.zero_()
        self.val_area_union.zero_()
        self.val_area_pred.zero_()
        self.val_area_label.zero_()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        images: Tensor = batch["image"]
        masks: Tensor = batch["mask"]
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = logits.argmax(dim=1)
        area_intersect, area_union, area_pred, area_label = (
            calculate_intersect_and_union(
                preds,
                masks,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
        )

        self.val_area_intersect += area_intersect.to(torch.float64)
        self.val_area_union += area_union.to(torch.float64)
        self.val_area_pred += area_pred.to(torch.float64)
        self.val_area_label += area_label.to(torch.float64)

        self._maybe_log_wandb_masks(
            split="val",
            images=images,
            masks=masks,
            preds=preds,
            batch=batch,
            batch_idx=batch_idx,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )
        return loss

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        area_intersect = self.all_gather(self.val_area_intersect)
        area_union = self.all_gather(self.val_area_union)
        area_label = self.all_gather(self.val_area_label)

        if area_intersect.dim() == 2:
            area_intersect = area_intersect.sum(dim=0)
            area_union = area_union.sum(dim=0)
            area_label = area_label.sum(dim=0)

        iou, acc, aacc = compute_iou_and_acc(area_intersect, area_union, area_label)
        miou = iou.nanmean()

        self.log(
            "val/mIoU",
            miou,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        self.log(
            "val/aAcc",
            aacc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )

        if self.trainer.is_global_zero:
            import wandb

            table = wandb.Table(columns=["class_id", "class_name", "iou"])
            iou_cpu = iou.detach().float().cpu()
            for class_id in range(self.num_classes):
                table.add_data(
                    class_id, CLASS_NAME_MAPPING[class_id], float(iou_cpu[class_id])
                )
            self.logger.experiment.log(
                {"val/per_class_iou": table}, step=self.global_step
            )

    def _maybe_log_wandb_masks(
        self,
        *,
        split: str,
        images: Tensor,
        masks: Tensor,
        preds: Tensor,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not self.wandb_log_masks:
            return
        if not self.trainer.is_global_zero:
            return
        if (int(self.global_step) % int(self.wandb_log_interval)) != 0 and (
            split == "train"
        ):
            return
        if split == "val" and batch_idx != 0:
            return

        import wandb

        image = images[0].detach().float().cpu()
        mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
        image = (image * std + mean).clamp(0.0, 1.0)
        image = (image * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

        class_color_mapping = {
            class_id: color for color, class_id in COLOR_CLASS_MAPPING.items()
        }

        import numpy as np

        gt_mask = masks[0].detach().to(torch.int64).cpu().numpy()
        pred_mask = preds[0].detach().to(torch.int64).cpu().numpy()

        color_lut = np.zeros((256, 3), dtype=np.uint8)
        for class_id in range(self.num_classes):
            color_lut[class_id] = np.asarray(
                class_color_mapping[class_id], dtype=np.uint8
            )
        color_lut[self.ignore_index] = np.asarray((255, 0, 255), dtype=np.uint8)
        gt_canvas = color_lut[gt_mask]
        pred_canvas = color_lut[pred_mask]

        alpha = 0.5
        image_f = image.astype(np.float32)
        gt_overlay = (
            (image_f * (1.0 - alpha) + gt_canvas.astype(np.float32) * alpha)
            .round()
            .astype(np.uint8)
        )
        pred_overlay = (
            (image_f * (1.0 - alpha) + pred_canvas.astype(np.float32) * alpha)
            .round()
            .astype(np.uint8)
        )

        gt_boundary = np.zeros(gt_mask.shape, dtype=bool)
        pred_boundary = np.zeros(pred_mask.shape, dtype=bool)
        gt_boundary[:-1, :] |= gt_mask[:-1, :] != gt_mask[1:, :]
        gt_boundary[1:, :] |= gt_mask[:-1, :] != gt_mask[1:, :]
        gt_boundary[:, :-1] |= gt_mask[:, :-1] != gt_mask[:, 1:]
        gt_boundary[:, 1:] |= gt_mask[:, :-1] != gt_mask[:, 1:]
        pred_boundary[:-1, :] |= pred_mask[:-1, :] != pred_mask[1:, :]
        pred_boundary[1:, :] |= pred_mask[:-1, :] != pred_mask[1:, :]
        pred_boundary[:, :-1] |= pred_mask[:, :-1] != pred_mask[:, 1:]
        pred_boundary[:, 1:] |= pred_mask[:, :-1] != pred_mask[:, 1:]

        boundary_color = np.asarray((255, 255, 255), dtype=np.uint8)
        gt_overlay[gt_boundary] = boundary_color
        pred_overlay[pred_boundary] = boundary_color

        caption = f"{split} step={int(self.global_step)} image={batch['image_path'][0]} mask={batch['mask_path'][0]}"
        self.logger.experiment.log(
            {
                f"{split}/overlay_ground_truth": wandb.Image(
                    gt_overlay, caption=caption
                ),
                f"{split}/overlay_prediction": wandb.Image(
                    pred_overlay, caption=caption
                ),
            },
            step=self.global_step,
        )

    def configure_optimizers(self):  # type: ignore[override]
        params = list(self.head.parameters())
        if not self.freeze_backbone:
            params = list(self.backbone.parameters()) + params
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.optimizer_cfg.lr),
            betas=(float(self.optimizer_cfg.beta1), float(self.optimizer_cfg.beta2)),
            weight_decay=float(self.optimizer_cfg.weight_decay),
        )

        if self.scheduler_cfg.type.lower() == "none":
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)

        if self.scheduler_cfg.type.lower() == "onecycle":
            warmup_iters = int(total_steps * float(self.scheduler_cfg.warmup_fraction))
            pct_start = float(warmup_iters) / float(total_steps)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.optimizer_cfg.lr),
                total_steps=total_steps,
                pct_start=pct_start,
                final_div_factor=float(self.scheduler_cfg.final_div_factor),
                anneal_strategy="cos",
                base_momentum=float(self.scheduler_cfg.base_momentum),
                max_momentum=float(self.scheduler_cfg.max_momentum),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "onecycle",
                },
            }

        if self.scheduler_cfg.type.lower() == "warmup_cosine":
            warmup_steps = int(total_steps * float(self.scheduler_cfg.warmup_fraction))

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = float(step - warmup_steps) / float(
                    total_steps - warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "warmup_cosine",
                },
            }

        raise ValueError(f"Unsupported scheduler.type: {self.scheduler_cfg.type}")


DINOv3LinearProbeSegModule = LinearProbeSegModule
