"""Autoresearch loop driver (DESIGN s7).

Per iteration:
  1. Codex proposes an edit to feature_construction.py / training.py (hypothesis + files).
  2. Apply edit to a sandbox copy; run leakage suite. If it fails -> revert, log.
  3. Build features (train) -> PPO train -> backtest on VALIDATION -> metrics.
  4. Keep if profit_score improved over best-so-far, else revert. Log either way.
  5. Repeat until iters/plateau. Best frozen.
  6. ONE-TIME eval on holdout (separate entrypoint: run_eval.py).

Deterministic harness: fixed seed; only the edited code changes between iterations.

This module ORCHESTRATES but does NOT auto-run on import. Use run_loop().
"""
from __future__ import annotations

import importlib
import shutil
import sys
import time
from pathlib import Path

from .experiment_log import ExperimentLog, LogEntry
from .game import load_split
from .optimizer import EDITABLE_FILES, CodeOptimizer

import subprocess

SRC = Path(__file__).resolve().parent
WORK = SRC.parent              # .autoresearch_nba (the dedicated git repo)
ARTIFACTS = WORK / "artifacts"
# Per-run output dir, set by run_loop() so each autoresearch run is self-contained
# and comparable: runs/<run_id>/ holds that run's metrics, experiment docs, log, best.
RUN_DIR = ARTIFACTS               # default; run_loop() overrides to runs/<run_id>/


def _read(name: str) -> str:
    return (SRC / name).read_text()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(WORK), *args],
                          capture_output=True, text=True).stdout.strip()


def _revert_to(sha: str) -> None:
    """Hard-restore the working tree to a known-good commit (drops the bad edit but
    keeps it in history as an experiment record)."""
    _git("reset", "--hard", sha)


def _write_dashboard_row(row: dict) -> None:
    """Append one iteration's metrics to THIS RUN's metrics.jsonl, and mirror to the
    stable artifacts/metrics.jsonl so the live dashboard always shows the current run."""
    import json as _json
    line = _json.dumps(row) + "\n"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "metrics.jsonl", "a") as f:
        f.write(line)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "metrics.jsonl", "a") as f:  # dashboard feed (latest run)
        f.write(line)


def _write_status(phase: str, it: int, iters: int, extra: dict | None = None) -> None:
    """Heartbeat so you can tell running-vs-stuck-vs-done. Written to the run dir AND
    artifacts/ (dashboard reads it). Includes a wall-clock stamp updated each phase."""
    import json as _json, time as _time
    st = {"phase": phase, "iter": it, "total_iters": iters,
          "updated_at": _time.strftime("%H:%M:%S"), "ts": _time.time()}
    if extra:
        st.update(extra)
    for d in (RUN_DIR, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(_json.dumps(st))


def _feature_snapshot() -> list[str]:
    """The live FEATURE_NAMES the current feature_construction produces — snapshotted
    each iteration so we can diff which features the agent added/removed."""
    import importlib, sys
    m = "src.feature_construction"
    if m in sys.modules:
        importlib.reload(sys.modules[m])
    from .feature_construction import FEATURE_NAMES
    return list(FEATURE_NAMES)


def _append_result(doc_path: str, kept: bool, metrics: dict, note: str,
                   cost_usd: float = 0.0, usage: dict | None = None) -> None:
    if not doc_path:
        return
    p = Path(doc_path)
    if not p.exists():
        return
    res = [f"\n## Result", f"- verdict: **{'KEPT' if kept else 'REVERTED'}** — {note}"]
    for k in ("profit_score", "mean_return", "sharpe", "win_rate", "avg_trades", "total_pnl"):
        if k in metrics:
            res.append(f"- {k}: {metrics[k]}")
    res.append(f"- codex_cost_usd: ${cost_usd:.4f}")
    if usage:
        res.append(f"- codex_tokens: in={usage.get('input_tokens',0)} "
                   f"(cached={usage.get('cached_input_tokens',0)}) "
                   f"out={usage.get('output_tokens',0)} "
                   f"reasoning={usage.get('reasoning_output_tokens',0)}")
    p.write_text(p.read_text() + "\n".join(res) + "\n")


def _reload_editable():
    """Reimport the editable modules so the harness uses the latest code."""
    for mod in ("feature_construction", "training", "evaluate", "leakage"):
        m = f"src.{mod}"
        if m in sys.modules:
            importlib.reload(sys.modules[m])


def _train_and_validate(train_games, val_games, seed: int, with_report: bool = False,
                        eval_games=None):
    _reload_editable()
    from .training import PPOConfig, train  # reimported
    from .evaluate import evaluate, score
    # capture the learning curves (train reward + periodic train/val PnL) so the
    # report can show under/overfitting to Codex.
    policy, critic, history = train(train_games, PPOConfig(seed=seed),
                                    val_games=val_games, return_history=True)
    metrics = evaluate(val_games, policy)          # VAL drives the keep decision
    # also measure train + eval (holdout) PnL for the dashboard's per-split view.
    # NOTE: this is display-only; the keep decision still uses VAL alone.
    splits = {"train_pnl": evaluate(train_games, policy).mean_return,
              "val_pnl": metrics.mean_return}
    if eval_games:
        splits["eval_pnl"] = evaluate(eval_games, policy).mean_return
    report = None
    if with_report:
        from .observability import build_report
        report = build_report(val_games, policy, critic, metrics.profit_score, score)
    return policy, critic, metrics, report, history, splits


def run_loop(data_dir: Path, iters: int = 20, seed: int = 0,
             model: str = "gpt-5.5", reasoning: str = "medium",
             trace: bool = True, run_id: str | None = None) -> None:
    from datetime import datetime
    from .tracing import TraceLogger
    from .dashboard import start_server
    global RUN_DIR
    # per-run folder so different autoresearch runs are self-contained & comparable
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = WORK / "runs" / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "EXPERIMENTS").mkdir(exist_ok=True)
    print(f"RUN DIR: runs/{run_id}/  (metrics, experiment docs, log, best for this run)")
    # reset the live dashboard feed so it shows ONLY this run
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "metrics.jsonl").write_text("")
    # live dashboard: non-fatal if a server already owns the port (e.g. run_dashboard.py)
    try:
        dash_url = start_server(6060)
        print(f"LIVE DASHBOARD: {dash_url}  (open in browser, refreshes every 2s)")
    except OSError:
        print("LIVE DASHBOARD: http://127.0.0.1:6060 (existing server reused)")
    log = ExperimentLog(RUN_DIR / "experiment_log.jsonl")
    tracer = TraceLogger(run_name=f"autoresearch-{model}-seed{seed}", enabled=trace)
    if tracer.active:
        print(f"tracing -> {tracer.endpoint} (open Raindrop Workshop / OTLP UI)")
    train_games = load_split(data_dir, "train")
    val_games = load_split(data_dir, "val")
    eval_games = load_split(data_dir, "eval")  # holdout — display-only per-split PnL
    print(f"loaded train={len(train_games)} val={len(val_games)} eval={len(eval_games)}")

    from dataclasses import asdict
    with tracer.span("run", {"iters": iters, "seed": seed, "model": model,
                             "n_train": len(train_games), "n_val": len(val_games)}):
        # baseline (iteration 0): current code, no edit. Full diagnostic report so the
        # optimizer can reason about HOW to improve, not just the profit_score.
        with tracer.span("iteration", {"iter": 0, "kind": "baseline"}) as sp:
            _t_base = time.time()
            _, _, base_m, base_report, base_hist, base_splits = _train_and_validate(train_games, val_games, seed, with_report=True, eval_games=eval_games)
            base_train_secs = round(time.time() - _t_base, 1)
            best_profit = base_m.profit_score
            best_files = {f: _read(f) for f in EDITABLE_FILES}
            cur_report = base_report
            sp.set_attrs(base_m.__dict__, prefix="metrics")
            sp.set(best_profit=best_profit, kept=True, verdict="baseline")
        total_cost = 0.0
        prev_feats = _feature_snapshot()
        base_row = dict(base_m.__dict__)
        base_row.update(iter=0, kept=True, best_profit=best_profit,
                        codex_cost_usd=0.0, total_cost_usd=0.0,
                        hypothesis="baseline (no agent edit)", commit="",
                        n_features=len(prev_feats), features=prev_feats,
                        features_added=[], features_removed=[],
                        train_secs=base_train_secs, iter_secs=base_train_secs,
                        train_pnl=base_splits.get("train_pnl"),
                        val_pnl=base_splits.get("val_pnl"),
                        eval_pnl=base_splits.get("eval_pnl"),
                        train_reward_curve=base_hist.get("train_reward", []),
                        train_pnl_curve=base_hist.get("train_pnl", []),
                        val_pnl_curve=base_hist.get("val_pnl", []))
        _write_dashboard_row(base_row)
        log.append(LogEntry(0, "baseline", log.file_hashes(best_files),
                            base_m.__dict__, True, best_profit, 0.0, "baseline"))
        print(f"[iter 0 baseline] profit_score={best_profit:.4f}")
        _write_status("baseline_done", 0, iters, {"best_profit": round(best_profit, 4)})

        opt = CodeOptimizer(model=model, reasoning_effort=reasoning)
        best_sha = _git("rev-parse", "HEAD")  # last KEPT commit (good tree)
        cur_hist = base_hist
        for it in range(1, iters + 1):
            t0 = time.time()
            diag = asdict(cur_report) if cur_report else {}
            if cur_hist:  # learning curves so Codex can see under/overfitting
                diag["train_reward_curve"] = cur_hist.get("train_reward", [])
                diag["train_pnl_curve"] = cur_hist.get("train_pnl", [])
                diag["val_pnl_curve"] = cur_hist.get("val_pnl", [])
            with tracer.span(f"iteration {it}", {"iter": it},
                             input=f"diagnostics + metrics (best profit_score={best_profit:.4f})") as sp:
                pre_sha = _git("rev-parse", "HEAD")
                kept = False; note = ""; metrics = {}; train_secs = 0.0; hist = {}; splits = {}
                prop = None; prior_error = None; iter_cost = 0.0; n_attempts = 0
                MAX_ATTEMPTS = 10  # retry (feeding the error back) until a clean trained run
                validated = False
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    phase = "codex_proposing" if attempt == 1 else f"codex_repair_{attempt-1}"
                    _write_status(phase, it, iters, {"best_profit": round(best_profit, 4)})
                    with tracer.span("codex.propose" if attempt == 1 else f"codex.repair{attempt-1}",
                                     {"model": model, "attempt": attempt}, kind="llm", model=model,
                                     input=(prior_error or f"diagnostics; best={best_profit:.4f}")[:500]) as psp:
                        prop = opt.propose(it, base_m.__dict__, {"profit_score": best_profit},
                                           diagnostics=diag, prior_error=prior_error)
                        psp.set_kind("llm", model=model, output=prop.hypothesis)
                        psp.set(hypothesis=prop.hypothesis, commit=prop.commit_sha[:8], attempt=attempt)
                    iter_cost += prop.cost_usd; n_attempts = attempt
                    if not prop.files:
                        prior_error = ("You made NO file changes. You MUST edit "
                                       "feature_construction.py and/or training.py this iteration.")
                        note = f"attempt {attempt}: no file changes -> retry"
                        _write_status("repair_needed", it, iters,
                                      {"attempt": attempt, "error": prior_error[:300]})
                        print(f"[iter {it}] attempt {attempt}: no changes -> retry", flush=True)
                        continue
                    # ---- ONLY GATE: training must run without error ----
                    try:
                        _write_status("training", it, iters, {"attempt": attempt})
                        with tracer.span("train_and_validate", kind="tool") as tsp:
                            _t_train = time.time()
                            _, _, m, report, hist, splits = _train_and_validate(train_games, val_games, seed, with_report=True, eval_games=eval_games)
                            train_secs = round(time.time() - _t_train, 1)
                            tsp.set_kind("tool", output=f"profit_score={m.profit_score:.4f} train_secs={train_secs}")
                            tsp.set(train_secs=train_secs); tsp.set_attrs(m.__dict__, prefix="metrics")
                        metrics = m.__dict__
                        # KEEP only if it beats best AND clears the risk-free hurdle:
                        # a policy must make REAL money (PnL above ~4% risk-free), not
                        # just barely-positive — else it's not worth deploying.
                        RISK_FREE = 0.04
                        if m.profit_score > best_profit and m.profit_score > RISK_FREE and m.avg_trades > 0:
                            best_profit = m.profit_score; best_sha = prop.commit_sha or best_sha
                            kept = True; note = f"KEPT (PnL/game {m.profit_score:.4f} > best & >4% hurdle)"
                            base_m = m; cur_report = report; cur_hist = hist
                        elif m.profit_score <= RISK_FREE:
                            note = f"reverted (PnL/game {m.profit_score:.4f} below 4% risk-free hurdle)"
                        else:
                            note = f"reverted (PnL/game {m.profit_score:.4f} <= best {best_profit:.4f})"
                        validated = True
                        break  # validated (kept or honestly-reverted) -> iteration done
                    except Exception as e:
                        # training crashed -> the ONLY failure mode -> feed exact error
                        # back to Codex and retry (visible in status + flushed log).
                        prior_error = f"Training raised {type(e).__name__}: {e}"
                        note = f"attempt {attempt} error: {type(e).__name__}: {str(e)[:60]}"
                        _write_status("repair_needed", it, iters,
                                      {"attempt": attempt, "error": prior_error[:300]})
                        print(f"[iter {it}] attempt {attempt}: {type(e).__name__} -> retry: {str(e)[:80]}", flush=True)
                        continue
                if not validated:
                    note = f"REJECTED: {MAX_ATTEMPTS} attempts could not produce a clean run — {note}"
                    print(f"[iter {it}] {note}")
                sp.set(hypothesis=prop.hypothesis if prop else "", commit=(prop.commit_sha[:8] if prop else ""))
                if prop and not prop.files and "no file changes" in note:
                    sp.set(verdict="no_change", kept=False)
                    log.append(LogEntry(it, prop.hypothesis, {}, {}, False, best_profit,
                                        time.time() - t0, note))
                    print(f"[iter {it}] {note}")
                    continue

                total_cost += iter_cost
                doc_path = prop.doc_path if prop else ""
                if kept:
                    _append_result(doc_path, kept, metrics, note, iter_cost, prop.usage)
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: result KEPT")
                else:
                    from pathlib import Path as _P
                    doc_txt = _P(doc_path).read_text() if doc_path and _P(doc_path).exists() else ""
                    _revert_to(pre_sha if pre_sha else best_sha)
                    if doc_txt and doc_path:
                        _P(doc_path).parent.mkdir(parents=True, exist_ok=True)
                        _P(doc_path).write_text(doc_txt)
                        _append_result(doc_path, kept, metrics, note, iter_cost, prop.usage if prop else {})
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: REVERTED — {note}")
                # snapshot the finished experiment doc into THIS run's folder
                if doc_path and Path(doc_path).exists():
                    (RUN_DIR / "EXPERIMENTS").mkdir(parents=True, exist_ok=True)
                    (RUN_DIR / "EXPERIMENTS" / Path(doc_path).name).write_text(
                        Path(doc_path).read_text())

                sp.set(verdict="kept" if kept else "reverted", kept=kept,
                       best_profit=best_profit, attempts=n_attempts,
                       profit_score=metrics.get("profit_score", 0.0) if metrics else 0.0,
                       codex_cost_usd=round(iter_cost, 4), total_cost_usd=round(total_cost, 4))
                # feature snapshot + diff vs the previous iteration (track feature changes)
                feats = _feature_snapshot()
                added = [f for f in feats if f not in prev_feats]
                removed = [f for f in prev_feats if f not in feats]
                prev_feats = feats
                # metrics row for the dashboard (always written, even on revert)
                metrics_row = dict(metrics)
                metrics_row.update(iter=it, kept=kept, best_profit=best_profit,
                                   codex_cost_usd=round(iter_cost, 4), total_cost_usd=round(total_cost, 4),
                                   attempts=n_attempts,
                                   hypothesis=(prop.hypothesis if prop else note), commit=(prop.commit_sha[:8] if prop else ""),
                                   n_features=len(feats), features=feats,
                                   features_added=added, features_removed=removed,
                                   train_secs=train_secs, iter_secs=round(time.time() - t0, 1),
                                   train_pnl=(splits or {}).get("train_pnl"),
                                   val_pnl=(splits or {}).get("val_pnl"),
                                   eval_pnl=(splits or {}).get("eval_pnl"),
                                   train_reward_curve=(hist or {}).get("train_reward", []),
                                   train_pnl_curve=(hist or {}).get("train_pnl", []),
                                   val_pnl_curve=(hist or {}).get("val_pnl", []))
                _write_dashboard_row(metrics_row)
                log.append(LogEntry(it, (prop.hypothesis if prop else note), {"commit": (prop.commit_sha if prop else "")},
                                    metrics_row, kept, best_profit, time.time() - t0, note))
                print(f"[iter {it}] {note} | {n_attempts} attempts | ${iter_cost:.4f} (cum ${total_cost:.2f}) | "
                      f"{(prop.commit_sha[:8] if prop else '')} | {(prop.hypothesis[:50] if prop else '')}")
    tracer.shutdown()
    _write_status("DONE", iters, iters, {"best_profit": round(best_profit, 4),
                                          "total_cost_usd": round(total_cost, 4)})
    (RUN_DIR / "best_profit.txt").write_text(str(best_profit))
    print(f"DONE. best val profit_score={best_profit:.4f} @ {best_sha[:8]} | total codex cost ${total_cost:.2f}")
