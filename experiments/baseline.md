# Baseline — the honest bar to beat

Dataset: 495 timestamped, leak-verified games (Feb 1 – Apr 30 2026).
Split (chronological, by game date): train 341 (Feb 1–Mar 24) / val 78 (Mar 25–Apr 4)
/ eval 76 (Apr 5–Apr 30). Reward = realized P&L in budget-fraction units (budget=1.0).
Leakage: 100% prefix-invariance pass on all games.

## Reference strategies (no learning, leak-free — entry uses only the tip-off candle)

| strategy | train | val | eval |
|---|---|---|---|
| always-skip (floor) | 0.0000 | 0.0000 | 0.0000 |
| buy-HOME-hold | −0.0245 | +0.0322 | +0.0895 |
| **buy-FAVORITE-hold** (the baseline) | **+0.0015** | **+0.0290** | +0.0831 |

## Why buy-FAVORITE-hold is THE baseline (not buy-HOME-hold)

buy-HOME-hold looks better on val/eval but is **negative on train** (−0.0245); it only
wins on val/eval because those samples skew home-friendly (home win rate 53% train →
58% val → 68% eval). Using it would be a lucky number, not a robust edge.

buy-FAVORITE-hold (only buy HOME when HOME is the tip-off favorite, hold to settle) is
the best simple strategy that is **positive on every split**. It is only *marginally*
positive on train (+0.0015/game) — there is no strong simple edge in this data.

## The bar

Any learned policy must beat **+0.0015/game on the TRAIN distribution it optimizes on**
(≈ +0.029 on val) to count as real edge. Read all three splits — a single split can
flatter or punish a strategy by luck of the home-win-rate draw.

Commit: `33786e2` (baseline harness).
