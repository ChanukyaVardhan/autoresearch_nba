#!/usr/bin/env python3
"""Quick demonstration: same trivial 'buy-and-hold' strategy under 4 sim configs.

Shows how the new friction layers (fees, slippage) compound across the val set.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest import BacktestEnv
from src.game import load_split
from src.types import Action


# Same strategy across all configs: buy ~50% at the first valid step, hold.
def buy_and_hold(env: BacktestEnv) -> None:
    # First action: BUY 0.5 of cash
    env.step(Action.BUY, 0.5)
    # Hold through the rest
    while not env.done:
        env.step(Action.SKIP_HOLD, 0.0)


CONFIGS = [
    ("perfect (old defaults)",
        dict(budget=100.0, fee_model="none", slippage_cents=0, latency=0, seed=0)),
    ("+ slippage 0–1¢",
        dict(budget=100.0, fee_model="none", slippage_cents=("uniform", 0, 1),
             latency=0, seed=0)),
    ("+ Kalshi fees",
        dict(budget=100.0, fee_model="kalshi", fee_rate_cents=1.75,
             slippage_cents=0, latency=0, seed=0)),
    ("realistic (fees + slippage 0–2¢)",
        dict(budget=100.0, fee_model="kalshi", fee_rate_cents=1.75,
             slippage_cents=("uniform", 0, 2), latency=0, seed=0)),
    ("stress (fees + slippage 1–3¢)",
        dict(budget=100.0, fee_model="kalshi", fee_rate_cents=1.75,
             slippage_cents=("uniform", 1, 3), latency=0, seed=0)),
]


@dataclass
class RunStats:
    label: str
    n_games: int
    mean_pnl: float
    median_pnl: float
    pnl_stdev: float
    win_rate: float
    total_pnl: float
    mean_pnl_pct: float


def run_config(games: list, label: str, env_kwargs: dict) -> RunStats:
    pnls = []
    for g in games:
        env = BacktestEnv(g, **env_kwargs)
        buy_and_hold(env)
        pnls.append(env.result().realized_pnl)
    arr = np.array(pnls, dtype=np.float64)
    return RunStats(
        label=label,
        n_games=len(games),
        mean_pnl=float(arr.mean()),
        median_pnl=float(np.median(arr)),
        pnl_stdev=float(arr.std()),
        win_rate=float((arr > 0).mean()),
        total_pnl=float(arr.sum()),
        mean_pnl_pct=float(arr.mean()) / env_kwargs["budget"] * 100,
    )


def main() -> None:
    DATA = Path(__file__).resolve().parent / "data"
    games = load_split(DATA, "val")
    print(f"loaded {len(games)} val games\n")

    rows: list[RunStats] = []
    for label, kw in CONFIGS:
        s = run_config(games, label, kw)
        rows.append(s)

    width = max(len(r.label) for r in rows)
    print(f"{'config'.ljust(width)}  {'mean':>9}  {'median':>9}  {'stdev':>9}  {'win%':>6}  {'mean%':>7}")
    print("-" * (width + 60))
    for r in rows:
        print(
            f"{r.label.ljust(width)}  "
            f"${r.mean_pnl:>7.3f}  "
            f"${r.median_pnl:>7.3f}  "
            f"${r.pnl_stdev:>7.3f}  "
            f"{r.win_rate*100:>5.1f}%  "
            f"{r.mean_pnl_pct:>6.2f}%"
        )
    print()
    # Show the marginal cost of each friction layer (delta from previous)
    print("marginal cost of each layer (vs. previous row):")
    for i in range(1, len(rows)):
        d = rows[i].mean_pnl - rows[i - 1].mean_pnl
        d_pct = (d / abs(rows[i - 1].mean_pnl) * 100) if rows[i - 1].mean_pnl else 0
        print(f"  {rows[i].label.ljust(width)}  ${d:+.3f}  ({d_pct:+.1f}%)")


if __name__ == "__main__":
    main()
