from __future__ import annotations

import torch
import torch.nn as nn


class DistTransformer(nn.Module):
    def __init__(
        self,
        window: int = 10,
        feat_dim: int = 8,
        d_model: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.window_len = 2 * window + 1
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.window_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        embeddings = self.input_proj(x) + self.pos_embed
        embeddings = self.transformer(embeddings)

        left = embeddings.unsqueeze(2).expand(
            batch_size, self.window_len, self.window_len, self.d_model
        )
        right = embeddings.unsqueeze(1).expand(
            batch_size, self.window_len, self.window_len, self.d_model
        )

        pair_features = torch.cat([left, right], dim=-1)
        distances = self.pair_mlp(pair_features.reshape(-1, 2 * self.d_model))
        distances = distances.view(batch_size, self.window_len, self.window_len)

        return 0.5 * (distances + distances.transpose(1, 2))


def build_model(
    window: int = 10,
    d_model: int = 64,
    nhead: int = 8,
    dim_feedforward: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> DistTransformer:
    return DistTransformer(
        window=window,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
    )
