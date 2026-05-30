# autoresearch_nba

Sequential-RL Kalshi NBA live-trading model + Codex (OpenAI) autoresearch loop.
Full design: `../DESIGN_autoresearch_trading.md`.

## Layout

```
data/                 raw pulled per-game data + split_manifest.csv (gitignored)
src/
  types.py            Action/Position/Candle/Score/Settlement dataclasses
  pbp_parser.py       PBP description parser + point-in-time box score + reconcile gate
  winprob.py          fixed score+time -> win-prob baseline (edge feature)
  game.py             Game object + loader (extraction/alignment, drops un-alignable games)
  feature_construction.py   *** CODEX-EDITABLE *** causal state encoder
  training.py               *** CODEX-EDITABLE *** PPO trainer
  backtest.py         FIXED trusted simulator (fills, +T clock, budget, settlement)
  networks.py         tiny numpy MLPs (policy + critic/sizer) + Adam
  evaluate.py         greedy rollout -> P&L curve + metrics + score()
  leakage.py          prefix-invariance + finite/dim tripwires (run every iter)
  optimizer.py        CodeOptimizer (OpenAI non-interactive); key from OPENAI_API_KEY
  experiment_log.py   append-only JSONL audit trail
  loop.py             autoresearch loop driver
run_extract.py        milestone 1: load+verify all games, freeze resolution artifacts
run_loop.py           run the Codex loop on validation
run_eval.py           ONE-TIME holdout eval
tests/                pbp parser + leakage tests
```

## Editable vs fixed
- **Codex edits:** `feature_construction.py`, `training.py` only.
- **Fixed/trusted (never edited):** `backtest.py`, `evaluate.py`, everything else.

## Run order
```sh
# 0) data already in data/ with split_manifest.csv (Feb1-Apr30, train/val/eval)
python3 -m pytest tests/ -q                 # parser + leakage unit tests
python3 run_extract.py                       # load/verify games, freeze resolutions
export OPENAI_API_KEY=sk-...                 # NEVER hardcode; rotate if leaked
python3 run_loop.py --iters 20               # autoresearch on validation
python3 run_eval.py                          # one-time holdout report
```

## Observability — Raindrop Workshop (live debugging UI)

Every autoresearch iteration emits an **OpenTelemetry trace** (`src/tracing.py`):
a `run` span containing one `iteration` span per loop turn, each with nested
`codex.propose`, `leakage_check`, and `train_and_validate` spans. Span attributes
carry the hypothesis, validation metrics, full diagnostics, and the keep/revert
verdict — so you can *watch the loop think* and filter/sort every experiment Codex
tries.

Raindrop Workshop is a **local CLI debugger** (no API key, no cloud) that consumes
OTLP traces. At the workshop:

```sh
curl -fsSL https://raindrop.sh/install | bash   # installs the raindrop CLI
raindrop workshop                                 # starts UI + OTLP collector @ :5899
# in another shell, run the loop pointed at Workshop:
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5899 python3 run_loop.py --iters 20
```

Tracing is **on by default** and **degrades to a no-op** if no OTLP endpoint is
reachable (so the loop never breaks). Disable with `python3 run_loop.py --no-trace`.
Endpoint precedence: `RAINDROP_LOCAL_DEBUGGER` > `OTEL_EXPORTER_OTLP_ENDPOINT`.
The local `artifacts/experiment_log.jsonl` is always written as the source-of-truth
record regardless of tracing.

## Hard rules (enforced by tests, not convention)
- `feature_construction` is pure + strictly causal (wall_clock <= t only).
- Game-end `player_stats` / settlement are reward/checksum ONLY — never features.
- Player-id resolution is frozen offline (deterministic); the loop only reads it.
- `wall_clock` may be a float string — parse as float.
