#!/usr/bin/env python3
"""Milestone 1: extraction/alignment. Load every game in split_manifest.csv into a
Game object, run the leakage suite + reconciliation, and report how many survive per
split. Writes a player-resolution artifact per game (name<->uuid), frozen offline.

Run from .autoresearch_nba/:  python3 run_extract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.game import load_game, load_split
from src.leakage import run_leakage_suite

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    import csv
    counts = {"train": [0, 0], "val": [0, 0], "eval": [0, 0]}  # [loaded, dropped]
    for r in csv.DictReader(open(DATA / "split_manifest.csv")):
        split = r["split"]
        if split not in counts:
            continue
        gdir = DATA / r["event_ticker"]
        g = load_game(gdir)
        if g is None:
            counts[split][1] += 1
            continue
        counts[split][0] += 1
        # freeze the resolution artifact
        (gdir / f"{r['event_ticker']}_player_id_resolution.json").write_text(
            json.dumps(g.name_to_uuid, indent=1)
        )
    print("split        loaded  dropped")
    for s, (lo, dr) in counts.items():
        print(f"{s:10s}  {lo:6d}  {dr:6d}")

    # leakage suite on a train sample
    train = load_split(DATA, "train")
    ok, msg = run_leakage_suite(train)
    print(f"\nleakage suite: {'PASS' if ok else 'FAIL'} — {msg}")


if __name__ == "__main__":
    main()
