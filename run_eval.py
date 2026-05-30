#!/usr/bin/env python3
"""ONE-TIME holdout eval (DESIGN s7 step 6). Trains on train+val with the FROZEN
best code, then reports metrics on the EVAL holdout. Touch the holdout once.

Run from .autoresearch_nba/:  python3 run_eval.py
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.game import load_split
from src.training import PPOConfig, train
from src.evaluate import evaluate

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    train_games = load_split(DATA, "train")
    val_games = load_split(DATA, "val")
    eval_games = load_split(DATA, "eval")
    print(f"train={len(train_games)} val={len(val_games)} eval={len(eval_games)}")

    # train on train+val with frozen best code, then eval ONCE on holdout
    policy, _ = train(train_games + val_games, PPOConfig(seed=0))
    m = evaluate(eval_games, policy)
    print("\n=== HOLDOUT (eval) METRICS ===")
    for k, v in asdict(m).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
