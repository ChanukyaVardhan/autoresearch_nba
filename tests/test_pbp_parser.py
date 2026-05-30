"""Unit tests for the load-bearing PBP description parser (DESIGN s4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pbp_parser import BoxScore, parse_event, reconcile


def test_made_shots_accumulate_points():
    b = BoxScore()
    parse_event(b, "twopointmade", "Andrew Wiggins makes two point jump shot")
    parse_event(b, "threepointmade", "Andrew Wiggins makes three point jump shot (Simone Fontecchio assists)")
    parse_event(b, "freethrowmade", "Andrew Wiggins makes regular free throw 1 of 2")
    assert b.players["Andrew Wiggins"]["points"] == 6
    assert b.players["Andrew Wiggins"]["three_points_made"] == 1
    assert b.players["Simone Fontecchio"]["assists"] == 1


def test_team_turnover_not_attributed_to_player():
    b = BoxScore()
    parse_event(b, "turnover", "Bulls turnover (5-second violation)")
    assert "Bulls" not in b.players  # team token, not a player


def test_block_attributes_to_blocker():
    b = BoxScore()
    parse_event(b, "twopointmiss", "Andrew Wiggins blocks Nikola Vucevic's two point driving hook shot")
    assert b.players["Andrew Wiggins"]["blocks"] == 1
    assert b.players["Nikola Vucevic"]["field_goals_attempted"] == 1


def test_reconcile_matches_by_points():
    b = BoxScore()
    # two players, distinct points totals -> resolved by (points, rebounds) key
    for _ in range(10):
        parse_event(b, "twopointmade", "Player One makes two point jump shot")  # 20 pts
    for _ in range(5):
        parse_event(b, "threepointmade", "Player Two makes three point jump shot")  # 15 pts
    stats = {
        "uuid-A": {"points": 20, "rebounds": 0},
        "uuid-B": {"points": 15, "rebounds": 0},
    }
    m = reconcile(b, stats)
    assert m is not None
    assert m["Player One"] == "uuid-A"
    assert m["Player Two"] == "uuid-B"


def test_reconcile_tolerates_secondary_stat_noise():
    """Identity is (points, rebounds); steals/blocks noise must NOT drop the game."""
    b = BoxScore()
    for _ in range(10):
        parse_event(b, "twopointmade", "Player One makes two point jump shot")  # 20 pts
    # parser missed a steal vs the official line; still must resolve
    stats = {"uuid-A": {"points": 20, "rebounds": 0, "steals": 3}}
    m = reconcile(b, stats)
    assert m == {"Player One": "uuid-A"}


def test_reconcile_rejects_mismatch():
    b = BoxScore()
    parse_event(b, "twopointmade", "Player One makes two point jump shot")  # 2 pts
    stats = {"uuid-A": {"points": 99}}  # checksum fails
    assert reconcile(b, stats) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok {name}")
    print("ALL PASS")
