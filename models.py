"""Compact feedback-aware GRU; same architecture without feedback for ablation."""
import torch
from torch import nn


class NextBeat(nn.Module):
    def __init__(self, n_items, dim=48, feedback=True):
        super().__init__()
        self.feedback = feedback
        self.embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.signals = nn.Linear(6, dim, bias=False)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.output = nn.Embedding(n_items, dim)
        self.bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.embedding.weight, std=0.05)
        nn.init.normal_(self.output.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)

    def forward(self, items, features):
        x = self.embedding(items)
        if self.feedback:
            x = x + self.signals(features)
        x = x * (items != 0).unsqueeze(-1)
        _, state = self.gru(x)
        return state[-1]

    def scores(self, items, features):
        return self(items, features) @ self.output.weight.T + self.bias.weight.T
