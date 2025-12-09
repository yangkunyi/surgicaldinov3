1. (DONE) Create a full pipeline with PyTorch Lightning, W&B and Hydra YAML to evaluate the DINOv3 checkpoints under `eval/depth`.
   - Entry point: `eval/depth/train_scared_depth.py` (Hydra main)
   - Config: `eval/depth/configs/scared_depth.yaml`
   - Lightning module: `eval/depth/pl_module.py` (`DinoDPTDepthModule`)
2. (DONE) Use the DINOv3 backbone loaded in the `eval/depth/dinov3_example.py` style and `eval/depth/dpt.py` as the depth head.
   - Backbone is loaded via `torch.hub.load(cfg.repo_dir, cfg.hub_name, source="local", weights=...)` inside `DinoDPTDepthModule`.
   - Optional teacher checkpoint is loaded from `cfg.model.backbone.trained_checkpoint` with `backbone.` prefix stripped.
3. (DONE) Use the WebDataset shards under `data/SCARED/shards/shard-train-0000xxx.tar` for training and `data/SCARED/shards/shard-val-0000xxx.tar` for validation.
   - Data module: `eval/depth/data_scared.py` (`ScaredDepthDataModule`).
   - Configurable shard patterns via `data.train_shards` / `data.val_shards` in the Hydra config.
4. (DONE) Freeze the backbone and only train the depth head.
   - `DinoDPTDepthModule._freeze_backbone()` disables gradients and sets the backbone to eval mode; the optimizer only sees trainable head parameters.
5. (DONE) Use the given loss.
   - Loss implementation: `eval/depth/loss.py` (`SigLoss`).
   - Wired into the Lightning module via `LossConfig` and used in both `training_step` and `validation_step`.

Example training command from the project root (adjust paths as needed):

    python -m eval.depth.train_scared_depth \
        model.backbone.pretrained_weights_path=dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        model.backbone.trained_checkpoint=dinov3/results/test/eval/training_2499/teacher_checkpoint.pth \
        data.train_shards="data/SCARED/shards/shard-train-{000000..000068}.tar" \
        data.val_shards="data/SCARED/shards/shard-val-{000000..000009}.tar"
