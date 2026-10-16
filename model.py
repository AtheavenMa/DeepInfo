"""Small independently trained CNN used only for the bundled synthetic demo."""
from __future__ import annotations
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.block = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(8, 12, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 2))

    def forward(self, x):
        return self.head(self.block(self.stem(x)))
