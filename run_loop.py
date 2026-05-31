#!/usr/bin/env python3
"""Run the Codex autoresearch loop on the validation split (DESIGN s7).

Requires OPENAI_API_KEY in the environment (never hardcoded). Run from .autoresearch_nba/:
    OPENAI_API_KEY=sk-... python3 run_loop.py --iters 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.loop import run_loop

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning", default="medium",
                    help="codex reasoning effort: minimal|low|medium|high|xhigh")
    ap.add_argument("--no-trace", action="store_true",
                    help="disable OTel/Raindrop tracing (on by default)")
    ap.add_argument("--run-id", default=None, help="reuse an existing run folder")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing run (keep prior iters, skip baseline)")
    ap.add_argument("--start-iter", type=int, default=1,
                    help="iteration to start/continue from (with --resume)")
    args = ap.parse_args()
    run_loop(DATA, iters=args.iters, seed=args.seed, model=args.model,
             reasoning=args.reasoning, trace=not args.no_trace,
             run_id=args.run_id, resume=args.resume, start_iter=args.start_iter)


if __name__ == "__main__":
    main()
