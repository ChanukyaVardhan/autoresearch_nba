"""Autoresearch loop driver (DESIGN s7).

Per iteration:
  1. Codex proposes an edit to feature_construction.py / training.py (hypothesis + files).
  2. Apply edit to a sandbox copy; run leakage suite. If it fails -> revert, log.
  3. Build features (train) -> PPO train -> backtest on VALIDATION -> metrics.
  4. Keep if headline improved over best-so-far, else revert. Log either way.
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
from .leakage import run_leakage_suite
from .optimizer import EDITABLE_FILES, CodeOptimizer

import subprocess

SRC = Path(__file__).resolve().parent
WORK = SRC.parent              # .autoresearch_nba (the dedicated git repo)
ARTIFACTS = WORK / "artifacts"


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
    """Append one iteration's metrics to artifacts/metrics.jsonl (the dashboard feed)."""
    import json as _json
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "metrics.jsonl", "a") as f:
        f.write(_json.dumps(row) + "\n")


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
    for k in ("headline", "mean_return", "sharpe", "win_rate", "avg_trades", "total_pnl"):
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


def _train_and_validate(train_games, val_games, seed: int, with_report: bool = False):
    _reload_editable()
    from .training import PPOConfig, train  # reimported
    from .evaluate import evaluate, score
    # capture the learning curves (train reward + periodic train/val PnL) so the
    # report can show under/overfitting to Codex.
    policy, critic, history = train(train_games, PPOConfig(seed=seed),
                                    val_games=val_games, return_history=True)
    metrics = evaluate(val_games, policy)
    report = None
    if with_report:
        from .observability import build_report
        report = build_report(val_games, policy, critic, metrics.headline, score)
    return policy, critic, metrics, report, history


def run_loop(data_dir: Path, iters: int = 20, seed: int = 0,
             model: str = "gpt-5.5", reasoning: str = "medium",
             trace: bool = True) -> None:
    from .tracing import TraceLogger
    from .dashboard import start_server
    # live dashboard: open this URL and watch iterations stream in as the loop runs
    dash_url = start_server(6060)
    print(f"LIVE DASHBOARD: {dash_url}  (open in browser, refreshes every 2s)")
    log = ExperimentLog(ARTIFACTS / "experiment_log.jsonl")
    tracer = TraceLogger(run_name=f"autoresearch-{model}-seed{seed}", enabled=trace)
    if tracer.active:
        print(f"tracing -> {tracer.endpoint} (open Raindrop Workshop / OTLP UI)")
    train_games = load_split(data_dir, "train")
    val_games = load_split(data_dir, "val")
    print(f"loaded train={len(train_games)} val={len(val_games)}")

    from dataclasses import asdict
    with tracer.span("run", {"iters": iters, "seed": seed, "model": model,
                             "n_train": len(train_games), "n_val": len(val_games)}):
        # baseline (iteration 0): current code, no edit. Full diagnostic report so the
        # optimizer can reason about HOW to improve, not just the headline.
        with tracer.span("iteration", {"iter": 0, "kind": "baseline"}) as sp:
            ok, msg = run_leakage_suite(train_games)
            if not ok:
                raise RuntimeError(f"baseline leakage failure: {msg}")
            _t_base = time.time()
            _, _, base_m, base_report, base_hist = _train_and_validate(train_games, val_games, seed, with_report=True)
            base_train_secs = round(time.time() - _t_base, 1)
            best_headline = base_m.headline
            best_files = {f: _read(f) for f in EDITABLE_FILES}
            cur_report = base_report
            sp.set_attrs(base_m.__dict__, prefix="metrics")
            sp.set(best_headline=best_headline, kept=True, verdict="baseline")
        total_cost = 0.0
        prev_feats = _feature_snapshot()
        base_row = dict(base_m.__dict__)
        base_row.update(iter=0, kept=True, best_headline=best_headline,
                        codex_cost_usd=0.0, total_cost_usd=0.0,
                        hypothesis="baseline (no agent edit)", commit="",
                        n_features=len(prev_feats), features=prev_feats,
                        features_added=[], features_removed=[],
                        train_secs=base_train_secs, iter_secs=base_train_secs,
                        train_reward_curve=base_hist.get("train_reward", []),
                        train_pnl_curve=base_hist.get("train_pnl", []),
                        val_pnl_curve=base_hist.get("val_pnl", []))
        _write_dashboard_row(base_row)
        log.append(LogEntry(0, "baseline", log.file_hashes(best_files),
                            base_m.__dict__, True, best_headline, 0.0, "baseline"))
        print(f"[iter 0 baseline] headline={best_headline:.4f}")

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
                             input=f"diagnostics + metrics (best headline={best_headline:.4f})") as sp:
                pre_sha = _git("rev-parse", "HEAD")
                # Codex agent span rendered as an LLM call (model, prompt-in, hypothesis-out).
                with tracer.span("codex.propose", {"model": model}, kind="llm",
                                 model=model,
                                 input=f"notes={diag.get('notes')} action_mix={diag.get('action_mix')} "
                                       f"value_corr={diag.get('value_corr')}") as psp:
                    prop = opt.propose(it, base_m.__dict__, {"headline": best_headline}, diagnostics=diag)
                    psp.set_kind("llm", model=model, output=prop.hypothesis)
                    psp.set(hypothesis=prop.hypothesis, commit=prop.commit_sha[:8],
                            files_changed=list(prop.files.keys()))
                sp.set(hypothesis=prop.hypothesis, commit=prop.commit_sha[:8])
                if not prop.files:
                    sp.set(verdict="no_change", kept=False)
                    log.append(LogEntry(it, prop.hypothesis, {}, {}, False, best_headline,
                                        time.time() - t0, "agent made no file changes"))
                    print(f"[iter {it}] agent made no file changes")
                    continue

                kept = False; note = ""; metrics = {}; train_secs = 0.0
                try:
                    with tracer.span("leakage_check", kind="tool",
                                     input="prefix-invariance + finite/dim on train sample") as lsp:
                        ok, msg = run_leakage_suite(train_games)
                        lsp.set_kind("tool", output=("PASS: " + msg) if ok else ("FAIL: " + msg))
                        lsp.set(passed=ok, detail=msg)
                    # independent anti-cheating verifier (Claude reviews Codex's diff)
                    vok, vreason = (True, "skipped")
                    if ok:
                        with tracer.span("verifier", kind="tool",
                                         input="Claude reviews diff for reward-hacking/leakage") as vsp:
                            from .verifier import verify
                            vok, vreason = verify()
                            vsp.set_kind("tool", output=("ACCEPT: " + vreason) if vok else ("REJECT: " + vreason))
                            vsp.set(passed=vok, detail=vreason)
                    if not ok:
                        note = f"REJECTED (leakage): {msg}"
                    elif not vok:
                        note = f"REJECTED (verifier): {vreason}"
                    else:
                        with tracer.span("train_and_validate", kind="tool",
                                         input="PPO train on train split -> eval on val split") as tsp:
                            _t_train = time.time()
                            _, _, m, report, hist = _train_and_validate(train_games, val_games, seed, with_report=True)
                            train_secs = round(time.time() - _t_train, 1)
                            tsp.set_kind("tool", output=f"headline={m.headline:.4f} mean_return={m.mean_return:.4f} "
                                                        f"win_rate={m.win_rate:.2f} trades/g={m.avg_trades:.1f} "
                                                        f"train_secs={train_secs}")
                            tsp.set(train_secs=train_secs)
                            tsp.set_attrs(m.__dict__, prefix="metrics")
                            if report:
                                tsp.set_attrs(asdict(report), prefix="diag")
                        metrics = m.__dict__
                        if m.headline > best_headline:
                            best_headline = m.headline; best_sha = prop.commit_sha or best_sha
                            kept = True; note = f"KEPT (headline {m.headline:.4f})"
                            base_m = m; cur_report = report; cur_hist = hist
                        else:
                            note = f"reverted (headline {m.headline:.4f} <= {best_headline:.4f})"
                except Exception as e:
                    note = f"REJECTED (error): {type(e).__name__}: {e}"

                total_cost += prop.cost_usd
                if kept:
                    _append_result(prop.doc_path, kept, metrics, note, prop.cost_usd, prop.usage)
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: result KEPT")
                else:
                    from pathlib import Path as _P
                    doc_txt = _P(prop.doc_path).read_text() if prop.doc_path and _P(prop.doc_path).exists() else ""
                    _revert_to(pre_sha if pre_sha else best_sha)
                    if doc_txt and prop.doc_path:
                        _P(prop.doc_path).parent.mkdir(parents=True, exist_ok=True)
                        _P(prop.doc_path).write_text(doc_txt)
                        _append_result(prop.doc_path, kept, metrics, note, prop.cost_usd, prop.usage)
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: REVERTED — {note}")

                sp.set(verdict="kept" if kept else "reverted", kept=kept,
                       best_headline=best_headline,
                       headline=metrics.get("headline", 0.0) if metrics else 0.0,
                       codex_cost_usd=prop.cost_usd, total_cost_usd=round(total_cost, 4))
                # feature snapshot + diff vs the previous iteration (track feature changes)
                feats = _feature_snapshot()
                added = [f for f in feats if f not in prev_feats]
                removed = [f for f in prev_feats if f not in feats]
                prev_feats = feats
                # metrics row for the dashboard (always written, even on revert)
                metrics_row = dict(metrics)
                metrics_row.update(iter=it, kept=kept, best_headline=best_headline,
                                   codex_cost_usd=prop.cost_usd, total_cost_usd=round(total_cost, 4),
                                   hypothesis=prop.hypothesis, commit=prop.commit_sha[:8],
                                   n_features=len(feats), features=feats,
                                   features_added=added, features_removed=removed,
                                   train_secs=train_secs, iter_secs=round(time.time() - t0, 1),
                                   train_reward_curve=(hist or {}).get("train_reward", []),
                                   train_pnl_curve=(hist or {}).get("train_pnl", []),
                                   val_pnl_curve=(hist or {}).get("val_pnl", []))
                _write_dashboard_row(metrics_row)
                log.append(LogEntry(it, prop.hypothesis, {"commit": prop.commit_sha},
                                    metrics_row, kept, best_headline, time.time() - t0, note))
                print(f"[iter {it}] {note} | ${prop.cost_usd:.4f} (cum ${total_cost:.2f}) | "
                      f"{prop.commit_sha[:8]} | {prop.hypothesis[:50]}")
    tracer.shutdown()
    (ARTIFACTS / "best_headline.txt").write_text(str(best_headline))
    print(f"DONE. best val headline={best_headline:.4f} @ {best_sha[:8]} | total codex cost ${total_cost:.2f}")
