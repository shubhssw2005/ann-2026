"""
ANN Model — Efficient Residual MLP
Optimized for CPU training: ~180k params, fast convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # LayerNorm — no segfault, faster than BatchNorm on CPU
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class BTCANN(nn.Module):
    """
    Input(45) → 256 → ResBlock×2 → 128 → ResBlock×1 → 64 → 3
    ~180k params — fast on CPU, strong accuracy.
    """

    def __init__(self, n_features: int, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(128, dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
