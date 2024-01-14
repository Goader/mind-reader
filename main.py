import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger

import torch

from fmri_encoder.datamodule import NSDDataModule
from fmri_encoder.model import FMRITransformerEncoder
from fmri_encoder.task import EmbeddingAlignmentTask

logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    wandb_logger = WandbLogger(project='mind-reader')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.checkpoint_output_dir,
        filename=f'{wandb_logger.experiment.name}_epoch{{epoch:02d}}_val_loss{{val_loss:.3f}}',
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        # plugins=[SLURMEnvironment(auto_requeue=False)],
        # overfit_batches=4,
        # fast_dev_run=True,
        max_epochs=cfg.task.max_epochs,
        log_every_n_steps=cfg.task.log_every_n_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.task.precision,
        gradient_clip_val=cfg.task.gradient_clip_val,
        gradient_clip_algorithm=cfg.task.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.task.accumulate_grad_batches,
        enable_model_summary=True,
    )
    # if no checkpoint_path is passed, then it is None, thus the model will start from the very beginning
    trainer.fit(task, datamodule=datamodule, ckpt_path=cfg.task.checkpoint_path)


def evaluate(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    if cfg.model.checkpoint_path is None:
        raise ValueError('no checkpoint path has been passed')

    raise NotImplementedError()


def inference(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    if cfg.model.checkpoint_path is None:
        raise ValueError('no checkpoint path has been passed')

    raise NotImplementedError()


@hydra.main(config_path='fmri_encoder/conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    datamodule = NSDDataModule(cfg)
    model = FMRITransformerEncoder(cfg)
    task = EmbeddingAlignmentTask(cfg, model)

    print(f'Encoder parameters: {sum(p.numel() / 1e6 for p in model.parameters() if p.requires_grad):.2f}M')

    match cfg.stage:
        case 'train':
            train(cfg, datamodule, task)
        case 'evaluate':
            evaluate(cfg, datamodule, task)
        case 'inference':
            inference(cfg, datamodule, task)
        case _:
            raise ValueError(
                'unknown stage, can be either `train`, `evaluate` or `inference`'
            )


if __name__ == '__main__':
    main()
