# CholecSeg8k Semantic Segmentation Eval (DINOv3) — Requirements

**Task:** Create a complete PyTorch Lightning pipeline to **linear-probe** a **frozen DINOv3** backbone on the **CholecSeg8k** dataset for semantic segmentation.

## 1) Deliverables / Expected File Layout

Implement a clean, self-contained segmentation eval module under `eval/segmentation/`:

- Entry point (Hydra main): `eval/segmentation/train_cholecseg8k.py`
- Single-file Hydra YAML config: `eval/segmentation/configs/cholecseg8k.yaml`
- Lightning module: `eval/segmentation/pl_module.py` (model, loss, metrics, optimizer/scheduler)
- Lightning data module: `eval/segmentation/data_cholecseg8k.py` (dataset + dataloaders)
- Local, copied/ported helpers (do not import from external local paths):
  - `eval/segmentation/seg_head.py` (linear head + DINOv3 feature adapter)
  - `eval/segmentation/seg_loss.py` (CE/dice; based on `dinov3/dinov3/eval/segmentation/loss.py`)
  - `eval/segmentation/seg_metrics.py` (mIoU/global-acc; based on `dinov3/dinov3/eval/segmentation/metrics.py`)
  - (optional) `eval/segmentation/seg_inference.py` for sliding-window inference, if needed

The code must be runnable from the `eval/segmentation/` directory without relying on `sys.path` hacks.

## 2) Project Requirements & Frameworks

- **Frameworks:** PyTorch Lightning, Weights & Biases (W&B) for logging, Hydra for configuration.
- **Backbone:** DINOv3 backbone must be **frozen** (`requires_grad=False`) and run in `eval()` mode; only the segmentation head is trainable.
- **Structure:** Use a `LightningModule` for model/loss/metrics and a `LightningDataModule` for data.
- **No defensive coding:** Assume paths/data are correct; crash on bad assumptions.

## 3) Dataset: CholecSeg8k (Video-Level Split)

- **Paths:**
  - Images: `/bd_byta6000i0/users/dataset/MedicalImage/CholecSeg8k/raw/videoXX/videoXX_XXXXX/frame_XX_endo.png`
  - Masks: `/bd_byta6000i0/users/dataset/MedicalImage/CholecSeg8k/raw/videoXX/videoXX_XXXXX/frame_XX_endo_color_mask.png`
- **Pairing rule:** Each image must map to exactly one mask with the `_color_mask` suffix in the same folder.
- **Video-level split (mandatory):** Split by top-level `videoXX` directory only. All frames from the same `videoXX` must be exclusively in train or val.
  - Config must support deterministic splitting via either:
    - `data.val_videos: [video01, video02, ...]`, or
    - `data.val_ratio` + `data.split_seed` (videos are sorted then split).
- **Classes:** 13 classes (IDs `0..12`), with class names from `eval/segmentation/colormap.py`.
- **Color mapping:** Use the exact color-to-class conversion logic from `eval/segmentation/colormap.py` (unknown colors must fail).

## 4) Transforms

- **Normalization:** Use `IMAGENET_MEAN` / `IMAGENET_STD` from `eval/segmentation/colormap.py`.
- **Train transforms (configurable):** random resize/scale, random crop, horizontal flip, normalization.
- **Val transforms (configurable):** deterministic resize (and optional center crop), normalization.
- **Mask interpolation:** nearest-neighbor only.
- Use torchvision transforms v2.

## 5) Segmentation Head, Backbone Features, and Output Resolution

Do not import segmentation head/loss/metrics from `dinov3/dinov3/eval/segmentation` directly; port/copy the logic locally.

- **Head:** Implement the linear head behavior from `dinov3/dinov3/eval/segmentation/models/heads/linear_head.py`:
  - copy the dinov3/dinov3/eval/segmentation/models/heads/linear_head.py
- **DINOv3 integration:** Ensure the forward pass correctly handles DINOv3 outputs:
  - extract the correct patch-token tensor(s) from the DINOv3 forward (use the same token convention as the repo’s DINOv3 backbone)
  - reshape patch tokens into a spatial feature map based on patch size and input resolution
- **Logits size:** Upsample logits to match the ground-truth mask resolution before computing loss/metrics.

## 6) Loss, Optimizer, Scheduler

- **Loss:** Cross entropy (ignore index `255`) and optional dice, following `dinov3/dinov3/eval/segmentation/loss.py`.
- **Optimizer:** AdamW over head parameters only.
- **Scheduler:** configurable (e.g., warmup + cosine/onecycle); keep all scheduler params in the single YAML config.

## 7) Metrics (Validation) and W&B Logging

- Compute and log during validation:
  - **Mean IoU (mIoU)** over the 13 classes
  - **Global Accuracy** (aAcc)
- Metrics must be computed via intersect/union aggregation across the **entire validation epoch** (and across devices under DDP), based on `dinov3/dinov3/eval/segmentation/metrics.py`.
- Log per-class IoU (table) to W&B using class names from `eval/segmentation/colormap.py`.

## 8) Logging Directories, Run Naming, and Hydra Constraints

- **Hydra config:** MUST be a single YAML file; do not use multiple config groups; do not use Python dataclasses.
- **Required log structure:**
  - W&B: `logs/wandb`
  - Hydra: `logs/hydra`
  - Lightning: `logs/lightning`
- **Naming requirements (config-driven):**
  - `run_name` must exist in the YAML config.
  - The log naming must include `${oc.env:OAR_JOB_ID}`.
  - `wandb_run_name` must exist in the YAML config.

## 9) Acceptance Checklist

- Running from `eval/segmentation/` works (`python train_cholecseg8k.py ...`).
- Train/val split is video-level only (no leakage).
- Backbone is frozen and never optimized.
- Validation logs include mIoU and global accuracy in W&B.
- Logs land in `logs/wandb`, `logs/hydra`, and `logs/lightning`.
