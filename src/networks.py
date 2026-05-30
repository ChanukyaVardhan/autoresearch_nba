"""PyTorch policy + critic/sizer nets (DESIGN s3). Tiny MLPs, CPU, deterministic
given a seed. The .policy()/.value() interfaces return numpy so the FIXED modules
(evaluate.py, observability.py, backtest rollout) are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .types import N_ACTIONS

SIZE_BUCKETS = np.array([0.25, 0.5, 1.0], dtype=np.float32)
N_SIZE = len(SIZE_BUCKETS)


def _mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_out),
    )


class PolicyNet(nn.Module):
    """Action head (N_ACTIONS) + size head (N_SIZE buckets)."""

    def __init__(self, n_in: int, n_hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.body = _mlp(n_in, n_hidden, n_hidden)
        self.action = nn.Linear(n_hidden, N_ACTIONS)
        self.size = nn.Linear(n_hidden, N_SIZE)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h = self.body(x)
        a_logits = self.action(h)
        a_logits = a_logits.masked_fill(~mask, -1e9)
        s_logits = self.size(h)
        return a_logits, s_logits

    # numpy interface used by evaluate/observability/backtest rollout
    def policy(self, x_np: np.ndarray, mask_np: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(x_np, np.float32))
            mask = torch.from_numpy(np.asarray(mask_np, bool))
            a_logits, s_logits = self.forward(x, mask)
            a_probs = torch.softmax(a_logits, dim=-1).numpy()
            s_probs = torch.softmax(s_logits, dim=-1).numpy()
        return a_probs, s_probs, None


class CriticNet(nn.Module):
    def __init__(self, n_in: int, n_hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed + 1)
        self.v = _mlp(n_in, n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)

    def value(self, x_np: np.ndarray):
        with torch.no_grad():
            v = self.forward(torch.from_numpy(np.asarray(x_np, np.float32))).numpy()
        return v, None
