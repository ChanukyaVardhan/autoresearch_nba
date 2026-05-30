"""The Game object + loader (DESIGN milestone 1: extraction/alignment layer).

A Game is the trusted, pre-aligned, immutable per-side time series for ONE market
(HOME). It exposes strictly-causal accessors that feature_construction() calls.
Point-in-time player box scores are precomputed per minute-step here (deterministic,
offline), so feature_construction does cheap lookups, not parsing.

Window = game start (first PBP event) -> game end (last PBP event), 1-min wall-clock
steps. Candle ts_utc and PBP wall_clock are the same UTC epoch clock.
"""
from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pbp_parser import BoxScore, parse_event, reconcile
from .types import Candle, PlayerLine, PositionState, ScoreState, Settlement

MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5}
STEP_SECONDS = 60  # T = 1 wall-clock minute


def _f(x: object) -> float:
    """Parse a possibly-float-string / blank value to float (NaN-safe)."""
    if x is None or x == "":
        return float("nan")
    return float(x)


def _i(x: object) -> int:
    return int(float(x))  # wall_clock is sometimes "1769987622.0" (DESIGN gotcha)


@dataclass
class Step:
    """A single decision minute: aligned candle + game/player state as of t."""

    t: int
    candle: Candle
    score: ScoreState
    # top-K player lines per team + team rollups are built lazily from box_at.


class Game:
    """Immutable aligned game for the HOME side. All accessors are causal in t."""

    def __init__(
        self,
        event_ticker: str,
        home_code: str,
        away_code: str,
        candles: list[Candle],
        pbp_rows: list[dict],
        box_snapshots: dict[int, BoxScore],
        settlement: Settlement,
        name_to_uuid: dict[str, str],
    ) -> None:
        self.event_ticker = event_ticker
        self.home_code = home_code
        self.away_code = away_code
        self._candles = candles  # sorted by ts
        self._candle_ts = [c.ts for c in candles]
        self._pbp = pbp_rows     # chronological, each has int wall_clock
        self._pbp_ts = [r["_wc"] for r in pbp_rows]
        self._box = box_snapshots  # t -> BoxScore (precomputed at each grid step)
        self.settlement = settlement
        self.name_to_uuid = name_to_uuid
        self._feat_cache: dict[int, object] = {}  # static-feature cache (speed)

        self.t_start = self._pbp_ts[0]
        self.t_end = self._pbp_ts[-1]
        # Decision grid: every minute from first to last PBP event (inclusive end).
        self.steps_ts = list(range(self.t_start, self.t_end + 1, STEP_SECONDS))

    # ---- causal accessors (feature_construction may only use these) ----

    def candle_at(self, t: int) -> Candle:
        """Latest candle with ts <= t."""
        i = bisect.bisect_right(self._candle_ts, t) - 1
        i = max(0, min(i, len(self._candles) - 1))
        return self._candles[i]

    def candles_window(self, t: int, k: int) -> list[Candle]:
        """Up to the last k candles with ts <= t (oldest..newest)."""
        i = bisect.bisect_right(self._candle_ts, t)
        return self._candles[max(0, i - k):i]

    def score_at(self, t: int) -> ScoreState:
        return _score_from_pbp(self._pbp, self._pbp_ts, t)

    def box_at(self, t: int) -> BoxScore:
        """Precomputed point-in-time box score for a grid step t."""
        if t in self._box:
            return self._box[t]
        # nearest precomputed step <= t (accessors may be called off-grid in tests)
        keys = sorted(self._box)
        i = bisect.bisect_right(keys, t) - 1
        return self._box[keys[max(0, i)]] if keys else BoxScore()

    def settlement_price(self) -> float:
        return 1.0 if self.settlement.home_won else 0.0


# ----------------------------------------------------------------------------- #
#  Loader
# ----------------------------------------------------------------------------- #

def _score_from_pbp(pbp: list[dict], pbp_ts: list[int], t: int) -> ScoreState:
    i = bisect.bisect_right(pbp_ts, t) - 1
    i = max(0, min(i, len(pbp) - 1))
    r = pbp[i]
    period = int(float(r.get("period_number") or 0))
    psr = _f(r.get("game_clock_seconds_remaining"))
    gsr = _f(r.get("game_elapsed_seconds"))
    # game_secs_remaining ~ regulation 48*60 minus elapsed (approx; OT handled loosely)
    game_remaining = max(0.0, 48 * 60 - (gsr if gsr == gsr else 0.0))
    return ScoreState(
        home_points=int(float(r.get("home_points") or 0)),
        away_points=int(float(r.get("away_points") or 0)),
        period=period,
        period_secs_remaining=psr if psr == psr else 0.0,
        game_secs_remaining=game_remaining,
        home_has_possession=False,  # possession is a team UUID; filled by loader
        last_event_is_timeout=(r.get("event_type") in ("teamtimeout", "challengetimeout")),
    )


def load_game(game_dir: Path) -> Optional[Game]:
    """Build a Game from a raw per-game dir. Returns None if the game can't be
    aligned/reconciled (untimestamped PBP, or player-stat reconciliation fails) —
    such games are DROPPED per DESIGN.
    """
    et = game_dir.name
    m = re.match(r"KXNBAGAME-26[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})", et)
    if not m:
        return None
    away_code, home_code = m.group(1), m.group(2)

    # --- candles: HOME side only ---
    cand_path = game_dir / f"{et}_kalshi_candles_1min.csv"
    if not cand_path.exists():
        return None
    home_candles: list[Candle] = []
    from datetime import datetime, timezone
    for r in csv.DictReader(open(cand_path)):
        if not str(r.get("ticker", "")).endswith(f"-{home_code}"):
            continue
        ts = int(datetime.fromisoformat(r["ts_utc"]).timestamp())
        home_candles.append(Candle(
            ts=ts,
            price_open=_f(r["price_open"]), price_high=_f(r["price_high"]),
            price_low=_f(r["price_low"]), price_close=_f(r["price_close"]),
            price_mean=_f(r["price_mean"]), price_previous=_f(r["price_previous"]),
            yes_bid_close=_f(r["yes_bid_close"]), yes_ask_close=_f(r["yes_ask_close"]),
            volume=_f(r["volume"]), open_interest=_f(r["open_interest"]),
        ))
    home_candles.sort(key=lambda c: c.ts)
    if len(home_candles) < 2:
        return None

    # --- PBP: require wall_clock; chronological ascending ---
    pbp_path = game_dir / f"{et}_kalshi_pbp.csv"
    raw = list(csv.DictReader(open(pbp_path)))
    if not raw or "wall_clock" not in raw[0]:
        return None
    pbp: list[dict] = []
    for r in raw:
        wc = r.get("wall_clock")
        if wc is None or wc == "":
            continue
        rr = dict(r)
        rr["_wc"] = _i(wc)
        pbp.append(rr)
    if len(pbp) < 5:
        return None
    pbp.sort(key=lambda r: r["_wc"])
    pbp_ts = [r["_wc"] for r in pbp]

    # --- settlement (END-STATE; reward path only) ---
    live = json.load(open(game_dir / f"{et}_live_data_with_player_stats.json"))
    det = live["live_data"]["details"]
    md = json.load(open(game_dir / f"{et}_milestone_details.json"))["milestone"]["details"]
    # NOTE: det["winner"] is a UUID from a DIFFERENT namespace than home_team_id, so
    # comparing them is always False. Determine the HOME YES outcome from the FINAL
    # SCORE (ground truth, namespace-independent): HOME YES settles 1 iff home wins.
    home_final = int(det.get("home_points", 0))
    away_final = int(det.get("away_points", 0))
    if home_final == away_final:
        return None  # no tie games in NBA; if scores equal, data is bad -> drop
    home_won = home_final > away_final
    settlement = Settlement(home_won=home_won, home_final=home_final, away_final=away_final)
    player_stats = det.get("player_stats", {})

    # --- point-in-time box score: replay PBP, snapshot at each grid step ---
    t_start, t_end = pbp_ts[0], pbp_ts[-1]
    grid = list(range(t_start, t_end + 1, STEP_SECONDS))
    box = BoxScore()
    snapshots: dict[int, BoxScore] = {}
    ev_idx = 0
    import copy
    for t in grid:
        while ev_idx < len(pbp) and pbp_ts[ev_idx] <= t:
            r = pbp[ev_idx]
            parse_event(box, r.get("event_type", ""), r.get("description", ""))
            ev_idx += 1
        snapshots[t] = copy.deepcopy(box)
    # consume any remaining events for the final reconciliation box
    while ev_idx < len(pbp):
        r = pbp[ev_idx]
        parse_event(box, r.get("event_type", ""), r.get("description", ""))
        ev_idx += 1

    # --- offline reconciliation gate (deterministic resolution or DROP) ---
    name_to_uuid = reconcile(box, player_stats) if player_stats else {}
    if player_stats and name_to_uuid is None:
        return None  # failed checksum/resolution -> drop game

    return Game(
        event_ticker=et, home_code=home_code, away_code=away_code,
        candles=home_candles, pbp_rows=pbp, box_snapshots=snapshots,
        settlement=settlement, name_to_uuid=name_to_uuid or {},
    )


def load_split(data_dir: Path, split: str) -> list[Game]:
    """Load all games tagged `split` in split_manifest.csv that load cleanly."""
    manifest = data_dir / "split_manifest.csv"
    games: list[Game] = []
    for r in csv.DictReader(open(manifest)):
        if r["split"] != split:
            continue
        g = load_game(data_dir / r["event_ticker"])
        if g is not None:
            games.append(g)
    return games
