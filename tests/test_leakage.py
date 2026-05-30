"""Prefix-invariance leakage test against real loaded games (DESIGN s4 tripwire)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game import load_split
from src.leakage import check_finite_and_dim, check_prefix_invariance

DATA = Path(__file__).resolve().parents[1] / "data"


def test_prefix_invariance_on_train_sample():
    games = load_split(DATA, "train")[:5]
    assert games, "no train games loaded — run after data is in place"
    for g in games:
        ok, msg = check_finite_and_dim(g)
        assert ok, f"{g.event_ticker}: {msg}"
        ok, msg = check_prefix_invariance(g)
        assert ok, f"{g.event_ticker}: {msg}"


if __name__ == "__main__":
    test_prefix_invariance_on_train_sample()
    print("ALL PASS")
