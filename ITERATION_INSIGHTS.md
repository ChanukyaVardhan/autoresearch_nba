# Iteration-by-Iteration Insights — what the agent figured out

A live autoresearch run (Codex gpt-5.5). Each iteration: the agent reads diagnostics,
makes ONE focused change to the feature encoder or PPO trainer, the harness trains &
scores it on validation, and KEEPS it only if PnL improves. Numbers below are real.

**The arc:** 0.37 → 0.66 → 0.67 → (3 misses, all rejected) → **0.72 best**.
PnL = mean realized profit per game (normalized stake). Eval = untouched holdout.

---

### Iteration 0 — Baseline (no agent edit)
- **val 0.37 · eval 0.35 · 72 trades/game · 65% win**
- The starting RL policy. It *makes money but over-trades wildly* (72 trades/game) —
  churning the market. The agent's job: turn this into real, efficient edge.

### Iteration 1 — "Stop the over-trading" ✅ KEPT (val 0.37 → 0.66)
- **What it changed:** added a **pyramid-buy penalty** (cost for stacking more lots on
  an open position) + a **sizing-head entropy bonus** (explore *how much* to bet, not
  just whether).
- **Insight:** the biggest win came from *trading smarter, not more*. Trades dropped
  72 → 38 while PnL nearly **doubled**. The agent correctly diagnosed churn as the
  problem and attacked it directly. One attempt, ~$0.21.

### Iteration 2 — "See the real tradeable edge" ✅ KEPT (0.66 → 0.67)
- **What it changed:** added **`buy_edge` / `sell_edge`** features — the model's
  win-prob vs the *actual ask* (buy) and *bid* (sell), not the mid-price.
- **Insight:** gave the policy signals priced at the real fills it transacts against,
  so it sees true edge net of the spread. Trades dropped further (38 → 33), win rate
  up to 79%. Small PnL gain but a *cleaner* policy.

### Iterations 3–5 — The plateau (all ❌ REVERTED, best held at 0.67)
- **iter 3** (val 0.65): a tweak that scored *below* best → auto-reverted.
- **iter 4** (val 0.00): an edit **collapsed the policy to always-skip** — 0 trades, 0
  PnL on every split. The 4%-hurdle gate rejected it instantly. *The best was never
  lost — a bad experiment costs one iteration, not the run.*
- **iter 5** (val 0.63): fewer trades (27) but lower PnL → reverted.
- **Insight:** the agent hit a wall and three ideas in a row didn't beat the best. This
  is what honest search looks like — most experiments fail, and the harness correctly
  throws them away rather than fooling itself.

### Iteration 6 — Breakthrough ✅ KEPT (0.67 → **0.72**, new best)
- **What it changed:** added **momentum-of-edge** features — `net_buy_edge`/
  `net_hold_edge` (edge *minus the spread cost*), and `edge_delta_180` /
  `model_wp_delta_180` (how the edge and win-prob are *moving* over the last 3 minutes).
- **The twist:** attempt 1 had a bug (`feature dim 57 != 62` — added feature names but
  not values). The harness fed the exact error back; the agent **self-repaired** and
  attempt 2 succeeded — *and turned out to be the best policy yet.*
- **Insight:** spread-aware *and time-derivative* edge signals let the policy time
  entries on momentum. val 0.72, **eval 0.74** (generalizes — holdout even higher),
  82% win, 32 trades. It broke the plateau by giving the net a richer view of *when*
  edge is appearing or fading.

---

## The story to tell the crowd
1. The agent **diagnosed and fixed over-trading** itself (iter 1) — the single biggest lever.
2. It **engineered spread-aware, momentum-aware edge features** (iters 2, 6) — exactly
   what a quant would do, discovered autonomously.
3. It **survived its own failures**: a policy collapse and a code bug, both handled by
   the guardrails — the best result was never at risk.
4. Total human input: **zero edits.** Total cost through the breakthrough: **~$1.66.**
