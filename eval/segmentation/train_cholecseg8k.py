from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from data_cholecseg8k import CholecSeg8kDataModule
from pl_module import LinearProbeSegModule


@hydra.main(config_path="configs", config_name="cholecseg8k", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)
    logs_root = Path(cfg.paths.logs_root)
    (logs_root / "wandb").mkdir(parents=True, exist_ok=True)
    (logs_root / "hydra").mkdir(parents=True, exist_ok=True)
    (logs_root / "lightning").mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb_run_name,
        tags=list(cfg.wandb.tags),
        log_model=cfg.wandb.log_model,
        save_dir=str(logs_root / "wandb" / cfg.wandb_run_name),
        offline=bool(cfg.wandb.offline),
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    datamodule = CholecSeg8kDataModule(cfg.data)
    model = LinearProbeSegModule(
        backbone_cfg=cfg.model.backbone,
        head_cfg=cfg.model.head,
        loss_cfg=cfg.loss,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        num_classes=int(cfg.data.num_classes),
        wandb_log_masks=bool(cfg.wandb.log_masks),
        wandb_log_interval=int(cfg.wandb.log_interval),
    )

    trainer = instantiate(cfg.trainer, logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
