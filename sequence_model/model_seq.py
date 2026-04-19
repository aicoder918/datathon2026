"""GRU sequence encoder + SSL and target heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, T, H)
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return out


class AttentionPool(nn.Module):
    """Single-query attention over time. (B, T, H) -> (B, H)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.score(h)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # softmax over T
        return (h * weights).sum(dim=1)         # (B, H)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NextReturnHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = _MLP(hidden, 1, hidden // 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, H) -> (B, T)
        return self.mlp(h).squeeze(-1)


class ReconHead(nn.Module):
    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.mlp = _MLP(hidden, out_dim, hidden // 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, H) -> (B, T, out_dim)
        return self.mlp(h)


class TargetHead(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = _MLP(hidden, 1, hidden, dropout=dropout)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:  # (B, H) -> (B,)
        return self.mlp(h_last).squeeze(-1)


class SequenceModel(nn.Module):
    """Encoder with three heads. SSL uses next_ret_head + recon_head; supervised
    uses target_head on an attention-pooled vector over all 50 bars (plus
    next_ret_head as aux)."""

    def __init__(self, input_dim: int, bar_dim: int, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden=hidden,
                                       n_layers=n_layers, dropout=dropout)
        self.next_ret_head = NextReturnHead(hidden)
        self.recon_head = ReconHead(hidden, bar_dim)
        self.attn_pool = AttentionPool(hidden)
        self.target_head = TargetHead(hidden, dropout=dropout)
        self.hidden = hidden
        self.input_dim = input_dim
        self.bar_dim = bar_dim

    def forward_ssl(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.next_ret_head(h), self.recon_head(h)

    def forward_supervised(self, x: torch.Tensor):
        h = self.encoder(x)
        pooled = self.attn_pool(h)              # (B, H) over all 50 bars
        y_hat = self.target_head(pooled)
        next_ret = self.next_ret_head(h)
        return y_hat, next_ret
