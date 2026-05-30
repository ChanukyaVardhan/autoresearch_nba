"""Per-game decision trace — the verification view (your ask #2): see EXACTLY what
the system did to earn its PnL. For each game: every BUY/SELL/SETTLE with the minute,
game clock, score, and price, plus the final outcome and realized PnL.

Writes artifacts/decisions.json (consumed by the dashboard 'Decisions' tab) and can
print a human-readable trace. This is FIXED/trusted (not Codex-editable).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .backtest import BacktestEnv
from .feature_construction import feature_construction
from .game import Game
from .networks import PolicyNet, SIZE_BUCKETS
from .types import Action

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def trace_game(game: Game, policy: PolicyNet) -> dict:
    """Run the greedy policy over one game, capturing every decision with context."""
    env = BacktestEnv(game)
    while True:
        x = feature_construction(game, env.t, env.pos)[None, :]
        mask = env.action_mask()
        a_probs, s_probs, _ = policy.policy(x, mask[None, :])
        a = int(a_probs[0].argmax())
        sz = float(SIZE_BUCKETS[int(s_probs[0].argmax())])
        if env.step(a, sz).done:
            break
    res = env.result()

    # enrich each trade with score/clock context at its minute
    actions = []
    for tr in res.trades:
        sc = game.score_at(tr.t)
        actions.append({
            "minute": round((tr.t - game.t_start) / 60.0, 1),
            "action": tr.action,                 # BUY / SELL / SETTLE
            "price": round(tr.price, 3),
            "size": round(tr.size, 3),
            "home_pts": sc.home_points, "away_pts": sc.away_points,
            "margin": sc.home_points - sc.away_points,
            "period": sc.period,
        })
    return {
        "game": game.event_ticker,
        "home_won": game.settlement.home_won,
        "final": f"{game.settlement.home_final}-{game.settlement.away_final}",
        "realized_pnl": round(res.realized_pnl, 4),
        "n_buys": res.n_buys, "n_sells": res.n_sells,
        "n_actions": res.n_buys + res.n_sells,
        "outcome": "WIN" if res.realized_pnl > 0 else ("LOSS" if res.realized_pnl < 0 else "FLAT"),
        "actions": actions,
    }


def dump_decisions(games: list[Game], policy: PolicyNet, split: str = "val") -> Path:
    traces = [trace_game(g, policy) for g in games]
    # summary so the verifier view leads with the aggregate
    pnls = [t["realized_pnl"] for t in traces]
    summary = {
        "split": split,
        "n_games": len(traces),
        "total_pnl": round(sum(pnls), 4),
        "mean_pnl": round(sum(pnls) / max(1, len(pnls)), 4),
        "wins": sum(1 for p in pnls if p > 0),
        "losses": sum(1 for p in pnls if p < 0),
        "flat": sum(1 for p in pnls if p == 0),
        "win_rate": round(sum(1 for p in pnls if p > 0) / max(1, len(pnls)), 3),
        "total_actions": sum(t["n_actions"] for t in traces),
        "avg_actions_per_game": round(sum(t["n_actions"] for t in traces) / max(1, len(traces)), 2),
    }
    out = ARTIFACTS / "decisions.json"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # sort games worst->best so failures are easy to inspect
    traces.sort(key=lambda t: t["realized_pnl"])
    out.write_text(json.dumps({"summary": summary, "games": traces}, indent=1))
    return out


def print_trace(game_trace: dict) -> None:
    t = game_trace
    print(f"\n{t['game']}  final {t['final']}  home_won={t['home_won']}  "
          f"-> {t['outcome']} pnl={t['realized_pnl']:+.4f}  ({t['n_actions']} actions)")
    for a in t["actions"]:
        print(f"   min {a['minute']:5.1f} Q{a['period']} {a['margin']:+3d}  "
              f"{a['action']:6s} @ {a['price']:.2f} size {a['size']:.2f}")
