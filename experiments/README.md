# Experiment records

One file per experiment. The autoresearch loop (and humans) write these.

- `baseline.md` — the honest bar to beat (reference strategies, per split).
- `iter-NN.md` — one experiment each:
  - **Commit / Author / Model intent** — the git commit, who made it (harness or the
    Codex agent), and what the model intended to do.
  - **## Hypothesis** — one sentence: expected improvement + why.
  - **## Rationale** — which diagnostic motivated it; the mechanism.
  - **## Diff summary** — the concrete code changes.
  - **## Result** — filled in by the harness after grading: verdict (KEPT/REVERTED)
    + val headline / mean_return / win_rate / trades-per-game.

The bar (from baseline.md): beat buy-FAVORITE-hold ≈ +0.0015/game on TRAIN
(≈ +0.029 on val). Read all three splits — home-win-rate skew can flatter one split.
