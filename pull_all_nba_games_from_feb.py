#!/usr/bin/env python3
"""Pull all KXNBAGAME games in a date range via the events+milestones listing.

Unlike pull_all_nba_playoff_games.py, discovery here uses
list_events_response(with_milestones=True) instead of list_markets, because
list_markets is trimmed by the API to roughly the last ~9 weeks and silently
drops older games. The milestone listing reaches back to January, so this can
pull Feb 1 -> today. See DESIGN_autoresearch_trading.md Appendix A.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent      # repo root (where the autoresearch system lives)
DATA_DIR = THIS_DIR / "data"                    # raw pulled per-game data lands here
KALSHI_PULL_PATH = THIS_DIR / "kalshi_pull.py"
PLAYOFF_PULL_PATH = THIS_DIR / "pull_all_nba_playoff_games.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kalshi_pull = load_module(KALSHI_PULL_PATH, "kalshi_pull")
playoff_pull = load_module(PLAYOFF_PULL_PATH, "playoff_pull")  # reuse enrich/analyze helpers


def with_retry(fn, *args, attempts: int = 6, base_sleep: float = 1.0, **kwargs):
    """Retry on HTTP 429 (rate limit) with exponential backoff."""
    import requests
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429 and attempt < attempts - 1:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            raise

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull KXNBAGAME games by milestone discovery.")
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    parser.add_argument("--series-ticker", default="KXNBAGAME")
    parser.add_argument("--start-date", default="2026-02-01", help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="inclusive YYYY-MM-DD; default today UTC")
    # We only care about odds from just before tip-off to game end; the market
    # opens days earlier but that pre-game drift is irrelevant. Keep a tiny pre pad
    # only so include_latest_before_start can seed the opening price.
    parser.add_argument("--pre-game-minutes", type=int, default=2)
    parser.add_argument("--post-game-minutes", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between games (be polite; avoids 429)")
    return parser.parse_args()


def ticker_date(event_ticker: str) -> Optional[datetime]:
    """KXNBAGAME-26FEB14LALBOS -> date(2026-02-14)."""
    m = re.match(r"KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    if mon not in MONTHS:
        return None
    return datetime(2000 + int(yy), MONTHS[mon], int(dd), tzinfo=timezone.utc)


def discover_games(client, series_ticker: str) -> dict[str, str]:
    """Return {event_ticker -> milestone_id} from the milestone listing."""
    payload = client.list_events_response(series_ticker=series_ticker, with_milestones=True)
    mapping: dict[str, str] = {}
    for milestone in payload["milestones"]:
        milestone_id = milestone.get("id")
        if not isinstance(milestone_id, str):
            continue
        tickers: list[str] = []
        details = milestone.get("details")
        if isinstance(details, dict):
            main_ticker = details.get("main_game_event_ticker")
            if isinstance(main_ticker, str):
                tickers.append(main_ticker)
        for key in ("related_event_tickers", "primary_event_tickers"):
            values = milestone.get(key)
            if isinstance(values, list):
                tickers.extend(str(v) for v in values)
        for ticker in tickers:
            if ticker.startswith(f"{series_ticker}-") and ticker not in mapping:
                mapping[ticker] = milestone_id
    return mapping


def write_json(payload: object, out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def team_codes(event_ticker: str) -> tuple[str, str]:
    """KXNBAGAME-26FEB14LALBOS -> ('LAL','BOS')  (away, home)."""
    m = re.match(r"KXNBAGAME-\d{2}[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})", event_ticker)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def pull_game(client, series_ticker, event_ticker, milestone_id, out_dir, pre_min, post_min, overwrite):
    away_code, home_code = team_codes(event_ticker)
    game_dir = out_dir / event_ticker
    game_dir.mkdir(parents=True, exist_ok=True)
    prefix = game_dir / event_ticker

    pbp_path = prefix.with_name(f"{event_ticker}_kalshi_pbp.csv")
    stats_path = prefix.with_name(f"{event_ticker}_kalshi_game_stats.json")
    live_path = prefix.with_name(f"{event_ticker}_live_data_with_player_stats.json")
    candles_path = prefix.with_name(f"{event_ticker}_kalshi_candles_1min.csv")
    # milestone details carries away/home_team_id, player_id_mapping (UUID<->UUID),
    # and series_to_stat maps. No human names exist anywhere in the API — those live
    # only in the PBP description free-text (see DESIGN doc s4).
    milestone_path = prefix.with_name(f"{event_ticker}_milestone_details.json")

    status: dict[str, object] = {
        "event_ticker": event_ticker,
        "milestone_id": milestone_id,
        "away_code": away_code,
        "home_code": home_code,
        "game_dir": str(game_dir),
        "play_by_play_csv": str(pbp_path),
        "game_stats_json": str(stats_path),
        "candles_csv": str(candles_path),
        "milestone_details_json": str(milestone_path),
        "error": "",
    }

    # PBP + player stats by milestone id (works for old games)
    if overwrite or not stats_path.exists() or not pbp_path.exists():
        stats_payload = with_retry(client.get_game_stats, milestone_id)
        write_json(stats_payload, stats_path)
        pbp_rows = kalshi_pull.flatten_play_by_play(stats_payload)
        playoff_pull.enrich_play_by_play(pd.DataFrame(pbp_rows)).to_csv(pbp_path, index=False)
    if overwrite or not live_path.exists():
        try:
            live = with_retry(client.get_live_data, milestone_id, include_player_stats=True)
            write_json(live, live_path)
        except Exception as exc:  # live_data can be absent for some games
            status["live_error"] = repr(exc)
    if overwrite or not milestone_path.exists():
        try:
            md = with_retry(client.get_json, f"milestones/{milestone_id}", {})
            write_json(md, milestone_path)
        except Exception as exc:
            status["milestone_error"] = repr(exc)

    # Derive candle window from PBP wall-clock span. Some games come back from the
    # API with NO wall_clock on any event (column absent or all-null) — those have no
    # alignable timeline and are unusable for the point-in-time model, so skip them.
    pbp = pd.read_csv(pbp_path)
    if "wall_clock" not in pbp.columns:
        status["error"] = "pbp has no wall_clock column (API returned untimestamped PBP)"
        return status
    wc = pd.to_numeric(pbp["wall_clock"], errors="coerce").dropna()
    if wc.empty:
        status["error"] = "no wall_clock values in pbp (API returned untimestamped PBP)"
        return status
    # Game start = first PBP event; game end = last. Episode trades over this span;
    # tiny pad only to seed the opening price candle.
    game_start = int(wc.min())
    start_ts = game_start - pre_min * 60
    end_ts = int(wc.max()) + post_min * 60
    status["game_start_ts"] = game_start
    status["game_end_ts"] = int(wc.max())

    # Odds candles per side via historical endpoint, bounded window
    if overwrite or not candles_path.exists():
        markets_by_ticker: dict[str, dict[str, object]] = {}
        candles_by_market: dict[str, list[dict[str, object]]] = {}
        for code in (away_code, home_code):
            if not code:
                continue
            market_ticker = f"{event_ticker}-{code}"
            markets_by_ticker[market_ticker] = {
                "ticker": market_ticker,
                "event_ticker": event_ticker,
                "yes_sub_title": code,
            }
            candles_by_market[market_ticker] = with_retry(
                client.get_market_candles,
                series_ticker=series_ticker,
                market_ticker=market_ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
                include_latest_before_start=True,
                historical_fallback=True,
            )
        frame = kalshi_pull.candles_to_dataframe(candles_by_market, markets_by_ticker)
        frame.to_csv(candles_path, index=False)
        status["candle_rows"] = int(len(frame))
    return status


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    end = (
        datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
        if args.end_date
        else datetime.now(timezone.utc)
    )

    client = kalshi_pull.KalshiClient()
    discovered = discover_games(client, args.series_ticker)

    in_range: list[tuple[str, str, datetime]] = []
    for event_ticker, milestone_id in discovered.items():
        d = ticker_date(event_ticker)
        if d is None or not (start <= d <= end):
            continue
        in_range.append((event_ticker, milestone_id, d))
    in_range.sort(key=lambda x: x[2])
    if args.limit:
        in_range = in_range[: args.limit]
    print(f"Discovered {len(discovered)} games; {len(in_range)} in [{args.start_date}, {end.date()}]", flush=True)

    manifest_rows: list[dict[str, object]] = []
    for i, (event_ticker, milestone_id, d) in enumerate(in_range, start=1):
        print(f"[{i}/{len(in_range)}] {d.date()} {event_ticker}", flush=True)
        try:
            row = pull_game(
                client, args.series_ticker, event_ticker, milestone_id,
                out_dir, args.pre_game_minutes, args.post_game_minutes, args.overwrite,
            )
            row["game_date"] = d.date().isoformat()
            manifest_rows.append(row)
        except Exception as exc:
            manifest_rows.append({"event_ticker": event_ticker, "game_date": d.date().isoformat(), "error": repr(exc)})
            print(f"  error: {exc}", flush=True)
        time.sleep(args.sleep)

    manifest = pd.DataFrame(manifest_rows)
    # Split boundaries (regular-season only — playoff games Apr 16+ come from the API
    # with no per-event PBP wall_clock, so they can't be aligned and are excluded
    # from val/eval; see DESIGN doc s6). Chronological 66/18/14:
    #   train: Feb 1 - Mar 22 | val: Mar 23 - Apr 3 | eval: Apr 4 - Apr 15
    #   post-Apr-15: EXCLUDED (playoff stragglers, mostly untimestamped)
    def split_of(date_str: str) -> str:
        d = datetime.fromisoformat(date_str).date()
        if d <= datetime(2026, 3, 22).date():
            return "train"
        if d <= datetime(2026, 4, 3).date():
            return "val"
        if d <= datetime(2026, 4, 15).date():
            return "eval"
        return "excluded_playoff"
    if "game_date" in manifest:
        manifest["split"] = manifest["game_date"].map(split_of)
    manifest.to_csv(out_dir / "all_game_manifest.csv", index=False)
    print(f"Wrote manifest to {out_dir / 'all_game_manifest.csv'}")
    # The autoresearch loop reads data/split_manifest.csv (event_ticker + split).
    # Emit it here so `python3 run_extract.py` / `run_loop.py` work right after a pull.
    if {"event_ticker", "split"}.issubset(manifest.columns):
        split_manifest = manifest.loc[
            manifest["split"].isin(["train", "val", "eval"]), ["event_ticker", "split"]
        ]
        split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)
        print(f"Wrote split manifest to {out_dir / 'split_manifest.csv'}")
        print(manifest.groupby("split")["event_ticker"].nunique())


if __name__ == "__main__":
    main()
