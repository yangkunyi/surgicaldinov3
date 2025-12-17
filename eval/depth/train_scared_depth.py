"""Hydra + Lightning entrypoint for SCARED depth training.

This script wires together:

* ``DinoDPTDepthModule`` (frozen DINOv3 backbone + DPT depth head)
* ``ScaredDepthDataModule`` (WebDataset-based loader for SCARED)
* Weights & Biases logging via ``lightning.pytorch.loggers.WandbLogger``

Configuration is handled via Hydra; see ``configs/scared_depth.yaml`` for an
example. A typical run from the repository root looks like::

    python -m eval.depth.train_scared_depth \
        model.backbone.pretrained_weights_path=dinov3/checkpoints/your_pretrain.pth \
        model.backbone.trained_checkpoint=dinov3/results/1/eval/training_3999/teacher_checkpoint.pth
"""

from __future__ import annotations

import os

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from .data_scared import ScaredDepthDataModule
from .dataset import DepthDataModule
from .scared_lance import ScaredLanceDataModule
from .scared_lmdb import ScaredLmdbDataModule
from .pl_module import DinoDPTDepthModule
import sys



@hydra.main(config_path="configs", config_name="scared_depth", version_base=None)
def main(cfg: DictConfig) -> None:
    OAR_JOB_ID = os.getenv("OAR_JOB_ID", "000000")
    logs_root = cfg.logs_root
    # ---------------------- loguru configuration -------------------
    os.makedirs(os.path.join(logs_root, "loguru"), exist_ok=True)
    log_path = os.path.join(logs_root, "loguru", "train_{time}.log")
    # Avoid duplicate sinks if this is called multiple times (e.g. in tests)
    logger.remove()
    logger.add(log_path, level="INFO", enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stdout, level="INFO", enqueue=True, backtrace=True, diagnose=True)
    # Pretty-print the resolved config for quick inspection
    logger.info(
        "Starting SCARED depth training with config:\n{}", OmegaConf.to_yaml(cfg)
    )

    pl.seed_everything(cfg.seed)

    # ---------------------- build data module ----------------------
    backend = cfg.data.backend
    logger.info(f"Using data backend: {backend}")
    if backend == "webdataset":
        datamodule = ScaredDepthDataModule(cfg.data)
    elif backend == "hdf5":
        datamodule = DepthDataModule(cfg)
    elif backend == "lance":
        datamodule = ScaredLanceDataModule(cfg.data)
    elif backend == "lmdb":
        datamodule = ScaredLmdbDataModule(cfg.data)
    else:
        raise ValueError(
            f"Unsupported data.backend '{backend}', expected 'webdataset', 'hdf5', 'lance', or 'lmdb'."
        )

    # ---------------------- build model ----------------------------
    model = DinoDPTDepthModule(
        backbone_cfg=cfg.model.backbone,
        head_cfg=cfg.model.head,
        optim_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        loss_cfg=cfg.loss,
    )

    # ---------------------- W&B logger -----------------------------

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=f"{cfg.wandb.run_name}-{OAR_JOB_ID}",
        tags=list(cfg.wandb.tags),
        log_model=cfg.wandb.log_model,
        save_dir=os.path.join(logs_root, "wandb"),
    )
    # Useful to record the full hydra config as W&B config
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # ---------------------- Trainer -------------------------------
    # Trainer and its callbacks (e.g. TQDMProgressBar) are fully defined in the Hydra config.
    trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
    )
    logger.info("Trainer created. Starting fit().")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Trainer.fit() complete. Starting validate().")
    trainer.validate(model, datamodule=datamodule)
    logger.info("Validation complete. Exiting.")


if __name__ == "__main__":
    main()
