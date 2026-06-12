from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        widths = [32, 64, 128, hidden_dim]
        layers: list[nn.Module] = []
        current = in_channels
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(width),
                    nn.SiLU(inplace=True),
                ]
            )
            current = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).flatten(1)


class ACTPolicy(nn.Module):
    """Compact action-chunking transformer policy for CALVIN imitation learning."""

    def __init__(
        self,
        image_channels: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        task_vocab_size: int,
        hidden_dim: int,
        nheads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.image_encoder = ImageEncoder(image_channels, hidden_dim)
        self.state_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.SiLU())
        self.task_embed = nn.Embedding(task_vocab_size, hidden_dim)
        self.context_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.query_embed = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))

    def forward(self, image: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        task_index = task_index.clamp(min=0, max=self.task_embed.num_embeddings - 1)
        context = torch.cat(
            [self.image_encoder(image), self.state_proj(state), self.task_embed(task_index)], dim=-1
        )
        context = self.context_proj(context).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(image.size(0), -1, -1)
        tokens = torch.cat([context, queries], dim=1)
        encoded = self.transformer(tokens)[:, 1:]
        return self.action_head(encoded)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0) * pred.size(-1)
    return (torch.abs(pred - target) * weights).sum() / denom


def build_policy(config, use_wrist_image: bool, chunk_size: int) -> ACTPolicy:
    return ACTPolicy(
        image_channels=6 if use_wrist_image else 3,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        chunk_size=chunk_size,
        task_vocab_size=config.task_vocab_size,
        hidden_dim=config.hidden_dim,
        nheads=config.nheads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
