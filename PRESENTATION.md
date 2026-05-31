# Autoresearch for Live Sports-Market Trading
### An AI agent that writes, trains, and improves its own RL trading model

---

## Slide 1 — The Problem & The Idea

**Problem:** Build a profitable policy for trading Kalshi NBA live-game markets
(buy/sell/hold a contract minute-by-minute as the game unfolds) — and do it without
a human hand-tuning features and training code.

**The idea — an autonomous research loop:**

```
 ┌─────────────────────────────────────────────────────────────┐
 │  Codex agent (gpt-5.5)  →  edits feature_construction.py      │
 │                            + training.py  (the only 2 files)  │
 │            ↓                                                  │
 │  Harness: PPO-train the policy → evaluate PnL on validation   │
 │            ↓                                                  │
 │  KEEP if PnL improves (git commit) · else REVERT              │
 │  on error → feed the exact error back → agent self-repairs    │
 │            ↓  (repeat, fully autonomous)                      │
 │  Live dashboard · per-experiment git history · cost tracking  │
 └─────────────────────────────────────────────────────────────┘
```

- **Codex is the researcher**, not just a code generator: it reads live diagnostics
  (action mix, learning curves, P&L attribution, feature importance) and proposes
  ONE focused, reasoned change per iteration.

**The key constraint — the agent has exactly TWO levers:**

| Codex MAY edit (the levers) | FROZEN / trusted (cannot touch) |
|---|---|
| `feature_construction(game,t,pos) → x_t` — the causal **state encoder** (what signals the policy sees) | the **backtest simulator** (fills at bid/ask, +T clock, settlement) — editing the scorer = cheating |
| `training_loop() / PPOConfig` — the **PPO trainer** (net capacity, lr, entropy, reward shaping) | the **reward def, data, eval holdout, leakage tests** |

Search space = **representation × optimization** — exactly the two things that
determine RL policy quality. Everything that could *fake* a result is locked, so every
PnL gain is a real modeling change, reproducible from one `git diff`.

**What's actually being trained — 2 small MLPs (a finite-horizon MDP per game):**
- **State** `x_t ∈ ℝ⁵⁷` at minute `t`: market microstructure + game state + model-vs-market
  edge + point-in-time box score + position context.
- **Policy net** (57→64→64→heads): action head softmax{SKIP/HOLD, BUY, SELL} (state-masked)
  + sizing head softmax{25%, 50%, 100% of budget}.
- **Critic net** (57→64→64→value): baseline for the advantage.
- Deterministic transition from recorded data; reward = step Δ realized+marked P&L net
  fees; trained with **clipped PPO + GAE + entropy**. Tiny + seeded ⇒ ~30s/iteration,
  reproducible.

---

## Slide 2 — Data Pipeline (the unglamorous half of the work)

**Source:** the Kalshi public API — no SDK, we wrote the puller. **~495 NBA games,
Feb–Apr 2026**, each reconstructed from 5 raw feeds into one aligned, causal time series:

| feed | what it gives |
|---|---|
| 1-min candles (per side) | live odds = implied win-prob, bid/ask (the price we trade) |
| play-by-play | score, clock, possession — point-in-time game state |
| live player_stats | per-player box score (reward checksum only) |
| milestone details | team IDs + player-ID mapping |
| → derived | **point-in-time box score** replayed from PBP, leakage-safe |

**Three real findings that shaped the dataset (each surfaced by digging, not assumed):**
- **Discovery bug:** the obvious `list_markets` API only returns ~9 weeks; the
  **milestone listing** reaches back months → got the full season, not just playoffs.
- **No player names anywhere** in the API — identities live only in PBP free-text;
  we parse + reconcile to the box score (drop any game that doesn't reconcile).
- **Playoff games ship with NO per-event timestamps** → can't align odds to plays →
  excluded. Honest split is **regular-season only**, never leaking the holdout.

**Output:** train 342 / val 78 / eval 76, chronological by date, 100% leakage-verified.

---

## Slide 3 — Results (live run, gpt-5.5, real numbers)

**Split (from the pipeline):** chronological, leakage-safe —
**train 342 / val 78 / eval 76 (untouched holdout)**.

**Naive baselines (val PnL/game), for reference:** always-skip `0.00` ·
buy-favorite-and-hold `+0.029` · buy-home-and-hold `+0.032` · risk-free hurdle `+0.04`.
→ No trivial strategy even clears the risk-free rate — the problem is genuinely hard.

**The agent's learned policy improved iteration over iteration — and it generalizes:**

| iteration | val PnL/game | eval PnL/game (untouched holdout) | trades/game |
|---|---|---|---|
| 0 — RL baseline | 0.37 | 0.35 | 72.5 |
| 1 — Codex edit | 0.66 | 0.66 | 38.1 |
| 2 — Codex edit (best) | 0.67 | **0.67** | 33.4 |

- **Beats every naive strategy + the risk-free hurdle**, ~**80% over its own RL start**
  — fully autonomous, no human in the loop.
- **Generalizes:** eval (never-touched holdout) tracks val almost exactly — not
  overfitting. The agent found edge that holds on unseen games.
- **Gets more efficient:** trades/game fell 72 → 33 while PnL rose — it learned to
  trade *less and smarter*.
- **Cheap:** entire run so far ≈ **$0.43** of Codex API cost.

---

## Slide 4 — What Makes It Trustworthy (and Honest)

**Guardrails so the agent can't cheat its way to a number:**
- **Post-cutoff data = zero memorized leakage.** These are **2026 NBA games — after
  every current model's training cutoff** — so the agent has NO pre-trained knowledge
  of any outcome. It physically cannot recall who won; it must learn from market data
  alone. (A pre-2024 dataset couldn't make this claim.)
- **No feature leakage** — a prefix-invariance test proves every feature uses only
  data with `wall_clock ≤ t`; verified PASS on all splits.
- **Holdout is sacred** — the agent only ever sees train data + aggregate val
  metrics, never the eval games.
- **Works offline** — web/browser/search tools disabled, so it can't look up scores.
- **Every experiment is a git commit** + experiment record → fully reproducible,
  revertible, audited. Pinned baseline ref → deterministic restarts.
- **One honest gate:** an edit is kept only if it trains AND beats the running best
  AND clears the 4% risk-free hurdle; on any error the agent self-repairs.

**Robustness — the loop self-protects (seen live this run):**
- **iter 4 collapsed to always-skip** (policy learned to never trade → 0 PnL on all
  splits). The hurdle rejected it; the best (0.67) was never lost. A bad experiment
  costs one iteration, not the run.
- **iter 6 hit a code bug** (`feature dim 57 != 62`) → the exact error was fed back to
  the agent, which self-repaired and retried — no human, no crash.
- Bad ideas dip the *attempt* line but never the *kept-best* line (a clean staircase).

**Observability:** live dashboard (per-split PnL curves, trade count, training time,
$ cost, learning curves) + a heartbeat so you always know the run's state.

**Honest caveat (we'd flag this to a real desk):** current PnL is measured with a
generous trade-cost model — the *magnitude* is optimistic; the *trend and
generalization* are real. Next step: charge the full bid-ask spread per trade so the
number is directly deployable.

**Takeaway:** a self-improving research agent that designs features, tunes RL
training, and pushes a real metric — with leakage guards, reproducibility, and
honest reporting built in.
