"""Evaluation + scoring (DESIGN s5 metrics). Runs trained nets greedily over a
split's games and produces the P&L curve + risk-adjusted profit_score + guardrails.

This is the curve the autoresearch loop optimizes on VALIDATION, and the one-time
report on the EVAL holdout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .backtest import BacktestEnv
from .feature_construction import feature_construction
from .game import Game
from .networks import N_SIZE, PolicyNet, CriticNet, SIZE_BUCKETS
from .types import Action


@dataclass
class Metrics:
    n_games: int
    mean_return: float          # mean per-game realized P&L (budget-fraction units)
    std_return: float
    sharpe: float               # mean/std across games (risk-adjusted profit_score)
    win_rate: float             # fraction of games with positive P&L
    total_pnl: float
    avg_trades: float           # buys+sells per game (overtrading guard)
    avg_deployed: float         # peak fraction of budget deployed
    max_drawdown: float         # worst within-game drawdown, averaged
    profit_score: float = 0.0       # the number the loop maximizes


def _greedy_episode(game: Game, policy: PolicyNet) -> float:
    env = BacktestEnv(game)
    while True:
        x = feature_construction(game, env.t, env.pos)[None, :]
        mask = env.action_mask()
        a_probs, s_probs, _ = policy.policy(x, mask[None, :])
        a = int(np.argmax(a_probs[0]))
        sz = float(SIZE_BUCKETS[int(np.argmax(s_probs[0]))])
        tr = env.step(a, sz)
        if tr.done:
            break
    return env.result().realized_pnl


def evaluate(games: list[Game], policy: PolicyNet) -> Metrics:
    rets, trades, deployed, dds = [], [], [], []
    for g in games:
        env = BacktestEnv(g)
        while True:
            x = feature_construction(g, env.t, env.pos)[None, :]
            mask = env.action_mask()
            a_probs, s_probs, _ = policy.policy(x, mask[None, :])
            a = int(np.argmax(a_probs[0]))
            sz = float(SIZE_BUCKETS[int(np.argmax(s_probs[0]))])
            if env.step(a, sz).done:
                break
        r = env.result()
        rets.append(r.realized_pnl)
        trades.append(r.n_buys + r.n_sells)
        deployed.append(r.deployed_peak)
        dds.append(r.max_drawdown)
    rets = np.array(rets, np.float64)
    mean_r = float(rets.mean()) if len(rets) else 0.0
    std_r = float(rets.std()) if len(rets) else 0.0
    sharpe = mean_r / (std_r + 1e-9)
    m = Metrics(
        n_games=len(games),
        mean_return=mean_r,
        std_return=std_r,
        sharpe=sharpe,
        win_rate=float((rets > 0).mean()) if len(rets) else 0.0,
        total_pnl=float(rets.sum()),
        avg_trades=float(np.mean(trades)) if trades else 0.0,
        avg_deployed=float(np.mean(deployed)) if deployed else 0.0,
        max_drawdown=float(np.mean(dds)) if dds else 0.0,
    )
    m.profit_score = score(m)
    return m


def score(m: Metrics) -> float:
    """The loop maximizes PnL. That's it. profit_score == mean realized PnL per game."""
    return m.mean_return
