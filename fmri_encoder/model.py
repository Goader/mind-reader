from omegaconf import DictConfig

import torch
import torch.nn as nn


class FMRITransformerEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # we could use a relative positional embedding here, but it would need to happen on the attention level
        self.positional_embedding = nn.Embedding(cfg.model.max_seq_len, cfg.model.hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.hidden_size,
            nhead=cfg.model.num_attention_heads,
            dim_feedforward=cfg.model.dim_feedforward,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.model.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_embedding(torch.arange(x.size(1), device=x.device))
        x = self.encoder(x, is_causal=False)
        return x
