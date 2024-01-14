from typing import Dict, Any

from omegaconf import DictConfig

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info


class EmbeddingAlignmentTask(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = model
        self.projection = nn.Linear(cfg.model.hidden_size, cfg.task.target_hidden_size)

        """
        Potentially we could exploit another encoder model to embrace the Relaxed Contrastive Loss function
        which would force the learned embeddings to carry the semantic information of the input sequence
        by learning it from the other pre-trained encoder model.
        
        That would be mixed with MSE loss or another one to force the model to learn the embeddings.
        Hopefully, that would reduce the amount of data needed to train the model. 
        """
        self.loss = nn.MSELoss()

    def _common_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq, label_embeddings, labels_mask = batch
        model_output = self.model(input_seq)

        masked_model_output = model_output[labels_mask]
        model_embeddings = self.projection(masked_model_output)
        loss = self.loss(model_embeddings, label_embeddings[labels_mask])

        return model_embeddings, loss

    def training_step(self, batch, batch_idx):
        model_output, loss = self._common_step(batch)
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output, loss = self._common_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        model_output, loss = self._common_step(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        model_output, loss = self._common_step(batch)
        return model_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.task.learning_rate,
            weight_decay=self.hparams.task.weight_decay
        )
        return optimizer
