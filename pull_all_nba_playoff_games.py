#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
KALSHI_PULL_PATH = THIS_DIR / "kalshi_pull.py"
ANALYZE_GAME_PATH = THIS_DIR / "analyze_game_odds.py"
MPLCONFIGDIR = THIS_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kalshi_pull = load_module(KALSHI_PULL_PATH, "kalshi_pull")
analyze_game = load_module(ANALYZE_GAME_PATH, "analyze_game_odds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull Kalshi odds and play-by-play for completed NBA playoff games.")
    parser.add_argument("--out-dir", default=str(THIS_DIR / "all_games"))
    parser.add_argument("--series-ticker", default="KXNBAGAME")
    parser.add_argument("--min-close", default="2026-04-18")
    parser.add_argument("--pre-game-minutes", type=int, default=180)
    parser.add_argument("--post-game-minutes", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--event-ticker", action="append")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ts(value: object) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(str(value)).tz_convert("UTC") if pd.Timestamp(str(value)).tzinfo else pd.Timestamp(str(value), tz="UTC")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:90] or "game"


def event_sort_key(event_markets: pd.DataFrame) -> pd.Timestamp:
    close_time = parse_ts(event_markets["close_time"].dropna().iloc[0])
    if close_time is None:
        return pd.Timestamp.min.tz_localize("UTC")
    return close_time


def event_metadata(event_ticker: str, event_markets: pd.DataFrame) -> dict[str, object]:
    title = str(event_markets["title"].dropna().iloc[0])
    away_team, home_team = analyze_game.infer_teams(title)
    winner_rows = event_markets.loc[event_markets["result"] == "yes"]
    winner_team = str(winner_rows["yes_sub_title"].iloc[0]) if not winner_rows.empty else ""
    return {
        "event_ticker": event_ticker,
        "title": title,
        "away_team": away_team,
        "home_team": home_team,
        "winner_team": winner_team,
        "open_time": event_markets["open_time"].dropna().iloc[0] if "open_time" in event_markets else None,
        "close_time": event_markets["close_time"].dropna().iloc[0] if "close_time" in event_markets else None,
    }


def milestone_map(client, series_ticker: str) -> dict[str, str]:
    payload = client.list_events_response(series_ticker=series_ticker, with_milestones=True)
    mapping: dict[str, str] = {}
    for milestone in payload["milestones"]:
        milestone_id = milestone.get("id")
        if not isinstance(milestone_id, str):
            continue
        tickers: list[str] = []
        for key in ["related_event_tickers", "primary_event_tickers"]:
            values = milestone.get(key)
            if isinstance(values, list):
                tickers.extend(str(value) for value in values)
        details = milestone.get("details")
        if isinstance(details, dict):
            main_ticker = details.get("main_game_event_ticker")
            if isinstance(main_ticker, str):
                tickers.append(main_ticker)
        for ticker in tickers:
            mapping[ticker] = milestone_id
    return mapping


def write_json(payload: object, out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def clock_to_seconds(clock: object) -> Optional[float]:
    if clock is None or pd.isna(clock):
        return None
    parts = str(clock).split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return None


def period_sequence(period_type: object, period_number: object) -> Optional[int]:
    if period_number is None or pd.isna(period_number):
        return None
    number = int(period_number)
    if str(period_type) == "overtime":
        return 4 + number
    if str(period_type) == "quarter":
        return number
    return number


def period_length_seconds(period_type: object) -> int:
    return 5 * 60 if str(period_type) == "overtime" else 12 * 60


def elapsed_seconds_before_period(sequence: Optional[int]) -> Optional[int]:
    if sequence is None:
        return None
    if sequence <= 4:
        return (sequence - 1) * 12 * 60
    return 4 * 12 * 60 + (sequence - 5) * 5 * 60


def enrich_play_by_play(frame: pd.DataFrame) -> pd.DataFrame:
    derived_columns = [
        "pbp_source_row",
        "has_wall_clock",
        "event_ts_utc",
        "period_sequence",
        "game_clock_seconds_remaining",
        "game_elapsed_seconds",
        "game_elapsed_minutes",
        "pbp_chronological_index",
    ]
    enriched = frame.drop(columns=[column for column in derived_columns if column in frame.columns]).copy()
    enriched["pbp_source_row"] = range(len(enriched))
    if "wall_clock" in enriched:
        enriched["has_wall_clock"] = enriched["wall_clock"].notna()
        enriched["event_ts_utc"] = pd.to_datetime(enriched["wall_clock"], unit="s", utc=True, errors="coerce")
    else:
        enriched["has_wall_clock"] = False
        enriched["event_ts_utc"] = pd.NaT

    enriched["period_sequence"] = [
        period_sequence(period_type, period_number)
        for period_type, period_number in zip(enriched.get("period_type", []), enriched.get("period_number", []))
    ]
    enriched["game_clock_seconds_remaining"] = [clock_to_seconds(clock) for clock in enriched.get("clock", [])]
    elapsed_values: list[Optional[float]] = []
    for _, row in enriched.iterrows():
        sequence = row["period_sequence"]
        clock_remaining = row["game_clock_seconds_remaining"]
        if pd.isna(sequence) or pd.isna(clock_remaining):
            elapsed_values.append(None)
            continue
        before_period = elapsed_seconds_before_period(int(sequence))
        if before_period is None:
            elapsed_values.append(None)
            continue
        elapsed_values.append(before_period + period_length_seconds(row.get("period_type")) - float(clock_remaining))
    enriched["game_elapsed_seconds"] = elapsed_values
    enriched["game_elapsed_minutes"] = enriched["game_elapsed_seconds"] / 60

    sortable = enriched.sort_values(
        ["game_elapsed_seconds", "pbp_source_row"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    sortable["pbp_chronological_index"] = range(1, len(sortable) + 1)
    enriched = enriched.merge(
        sortable[["pbp_source_row", "pbp_chronological_index"]],
        on="pbp_source_row",
        how="left",
    )
    return enriched.sort_values("pbp_chronological_index").reset_index(drop=True)


def game_window_from_score(score: Optional[pd.DataFrame], fallback_close_time: pd.Timestamp, pre_minutes: int, post_minutes: int) -> tuple[int, int]:
    if score is not None and not score.empty:
        start = score["event_ts_utc"].min() - pd.Timedelta(minutes=pre_minutes)
        end = score["event_ts_utc"].max() + pd.Timedelta(minutes=post_minutes)
    else:
        start = fallback_close_time - pd.Timedelta(hours=6)
        end = fallback_close_time + pd.Timedelta(minutes=post_minutes)
    return int(start.timestamp()), int(end.timestamp())


def pull_game(
    client,
    series_ticker: str,
    event_ticker: str,
    event_markets: pd.DataFrame,
    milestone_ids: dict[str, str],
    out_dir: Path,
    pre_game_minutes: int,
    post_game_minutes: int,
    overwrite: bool,
) -> dict[str, object]:
    metadata = event_metadata(event_ticker, event_markets)
    game_slug = slugify(f"{event_ticker}_{metadata['title']}")
    game_dir = out_dir / game_slug
    game_dir.mkdir(parents=True, exist_ok=True)
    prefix = game_dir / game_slug

    markets_path = prefix.with_name(f"{game_slug}_kalshi_markets.csv")
    pbp_path = prefix.with_name(f"{game_slug}_kalshi_pbp.csv")
    stats_path = prefix.with_name(f"{game_slug}_kalshi_game_stats.json")
    candles_path = prefix.with_name(f"{game_slug}_kalshi_candles_1min_game_window.csv")
    summary_path = prefix.with_name(f"{game_slug}_signal_summary.csv")
    chart_path = prefix.with_name(f"{game_slug}_odds_score_chart.png")

    status: dict[str, object] = {
        **metadata,
        "game_dir": str(game_dir),
        "markets_csv": str(markets_path),
        "candles_csv": str(candles_path),
        "play_by_play_csv": str(pbp_path),
        "game_stats_json": str(stats_path),
        "summary_csv": str(summary_path),
        "chart_png": str(chart_path),
        "error": "",
    }

    event_markets.to_csv(markets_path, index=False)

    score = None
    if overwrite or not pbp_path.exists() or not stats_path.exists():
        milestone_id = milestone_ids.get(event_ticker)
        if milestone_id is None:
            status["error"] = "missing milestone"
            return status
        stats_payload = client.get_game_stats(milestone_id)
        write_json(stats_payload, stats_path)
        pbp_rows = kalshi_pull.flatten_play_by_play(stats_payload)
        enrich_play_by_play(pd.DataFrame(pbp_rows)).to_csv(pbp_path, index=False)
    elif pbp_path.exists():
        enrich_play_by_play(pd.read_csv(pbp_path)).to_csv(pbp_path, index=False)

    title = str(event_markets["title"].dropna().iloc[0])
    away_team, home_team = analyze_game.infer_teams(title)
    score = analyze_game.load_scoreboard(pbp_path, home_team, away_team)

    close_time = event_sort_key(event_markets)
    start_ts, end_ts = game_window_from_score(score, close_time, pre_game_minutes, post_game_minutes)

    if overwrite or not candles_path.exists():
        markets = event_markets.to_dict(orient="records")
        markets_by_ticker = {str(market["ticker"]): market for market in markets if market.get("ticker")}
        candles_by_market: dict[str, list[dict[str, object]]] = {}
        for ticker in markets_by_ticker:
            candles_by_market[ticker] = client.get_market_candles(
                series_ticker=series_ticker,
                market_ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
                include_latest_before_start=True,
                historical_fallback=True,
            )
        candles = kalshi_pull.candles_to_dataframe(candles_by_market, markets_by_ticker)
        candles.to_csv(candles_path, index=False)

    analyze_game.main_from_paths(candles_path, markets_path, pbp_path, prefix)
    status["probabilities_csv"] = str(prefix.with_name(f"{game_slug}_probabilities_wide.csv"))
    status["top_moves_csv"] = str(prefix.with_name(f"{game_slug}_top_odds_moves.csv"))
    return status


def select_events(markets: pd.DataFrame, event_tickers: Optional[list[str]], limit: Optional[int]) -> list[tuple[str, pd.DataFrame]]:
    finalized = markets.loc[markets["status"] == "finalized"].copy()
    if event_tickers:
        finalized = finalized.loc[finalized["event_ticker"].isin(event_tickers)]
    events = [(event_ticker, group.copy()) for event_ticker, group in finalized.groupby("event_ticker")]
    events.sort(key=lambda item: event_sort_key(item[1]))
    if limit is not None:
        return events[:limit]
    return events


def write_aggregate_outputs(out_dir: Path, manifest_rows: list[dict[str, object]]) -> None:
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "all_game_manifest.csv", index=False)

    summaries = []
    top_moves = []
    probabilities = []
    pbp = []
    candles = []
    for row in manifest_rows:
        if row.get("error"):
            continue
        event_ticker = str(row["event_ticker"])
        title = str(row["title"])
        for collection, path_key in [
            (summaries, "summary_csv"),
            (top_moves, "top_moves_csv"),
            (probabilities, "probabilities_csv"),
            (pbp, "play_by_play_csv"),
            (candles, "candles_csv"),
        ]:
            path_value = row.get(path_key)
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["event_ticker"] = event_ticker
            frame["event_title"] = title
            leading_columns = ["event_ticker", "event_title"]
            frame = frame[leading_columns + [column for column in frame.columns if column not in leading_columns]]
            collection.append(frame)

    aggregate_targets = [
        (summaries, "all_game_signal_summary.csv"),
        (top_moves, "all_game_top_odds_moves.csv"),
        (probabilities, "all_game_probabilities_wide_stacked.csv"),
        (pbp, "all_game_play_by_play.csv"),
        (candles, "all_game_candles_1min_game_window.csv"),
    ]
    for frames, filename in aggregate_targets:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = kalshi_pull.KalshiClient()
    markets = client.list_markets(
        series_ticker=args.series_ticker,
        min_close_ts=kalshi_pull.parse_timestamp(args.min_close),
    )
    markets_frame = pd.json_normalize(markets)
    markets_index_path = out_dir / "kalshi_nba_game_markets_from_cutoff.csv"
    if args.refresh_index or args.overwrite or not markets_index_path.exists():
        markets_frame.to_csv(markets_index_path, index=False)

    events = select_events(markets_frame, args.event_ticker, args.limit)
    print(f"Selected {len(events)} completed event(s)")
    milestones = milestone_map(client, args.series_ticker)

    manifest_rows: list[dict[str, object]] = []
    for index, (event_ticker, event_markets) in enumerate(events, start=1):
        title = str(event_markets["title"].dropna().iloc[0])
        print(f"[{index}/{len(events)}] {event_ticker} {title}", flush=True)
        try:
            manifest_rows.append(
                pull_game(
                    client=client,
                    series_ticker=args.series_ticker,
                    event_ticker=event_ticker,
                    event_markets=event_markets,
                    milestone_ids=milestones,
                    out_dir=out_dir,
                    pre_game_minutes=args.pre_game_minutes,
                    post_game_minutes=args.post_game_minutes,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:
            metadata = event_metadata(event_ticker, event_markets)
            manifest_rows.append({**metadata, "error": repr(exc)})
            print(f"  error: {exc}", flush=True)

    write_aggregate_outputs(out_dir, manifest_rows)
    print(f"Wrote aggregate outputs to {out_dir}")


if __name__ == "__main__":
    main()
