#!/usr/bin/env python3
"""
Generic public Kalshi market-data puller.

Examples:
    python kalshi_pull.py series --search basketball --out series.csv
    python kalshi_pull.py markets --series-ticker KXNBAGAME --query "Boston Philadelphia" --out markets.csv
    python kalshi_pull.py candles --event-ticker KXNBAGAME-26APR24BOSPHI --start 2026-04-24T12:00:00Z --end 2026-04-25T02:00:00Z --out candles.csv --plot candles.png

Public market-data endpoints do not require authentication.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = "kalshi-public-data-puller/0.1"


class KalshiClient:
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_json(self, path: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=clean_params(params or {}), timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {url}")
        return payload

    def paginated(self, path: str, list_key: str, params: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
        cursor = ""
        rows: List[Dict[str, object]] = []
        while True:
            page_params = dict(params or {})
            if cursor:
                page_params["cursor"] = cursor
            payload = self.get_json(path, page_params)
            page_rows = payload.get(list_key, [])
            if not isinstance(page_rows, list):
                raise ValueError(f"Expected list at response key {list_key}")
            rows.extend(row for row in page_rows if isinstance(row, dict))
            cursor_value = payload.get("cursor")
            cursor = cursor_value if isinstance(cursor_value, str) else ""
            if not cursor:
                return rows

    def list_series(
        self,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        include_volume: bool = False,
    ) -> List[Dict[str, object]]:
        return self.paginated(
            "series",
            "series",
            {
                "category": category,
                "tags": tags,
                "include_volume": bool_param(include_volume),
            },
        )

    def list_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        with_nested_markets: bool = False,
    ) -> List[Dict[str, object]]:
        return self.paginated(
            "events",
            "events",
            {
                "series_ticker": series_ticker,
                "status": status,
                "min_close_ts": min_close_ts,
                "with_nested_markets": bool_param(with_nested_markets),
                "limit": 200,
            },
        )

    def list_events_response(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        with_milestones: bool = False,
    ) -> Dict[str, List[Dict[str, object]]]:
        cursor = ""
        events: List[Dict[str, object]] = []
        milestones: List[Dict[str, object]] = []
        while True:
            params = {
                "series_ticker": series_ticker,
                "status": status,
                "with_milestones": bool_param(with_milestones),
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self.get_json("events", params)
            page_events = payload.get("events", [])
            page_milestones = payload.get("milestones", [])
            if isinstance(page_events, list):
                events.extend(row for row in page_events if isinstance(row, dict))
            if isinstance(page_milestones, list):
                milestones.extend(row for row in page_milestones if isinstance(row, dict))
            cursor_value = payload.get("cursor")
            cursor = cursor_value if isinstance(cursor_value, str) else ""
            if not cursor:
                return {"events": events, "milestones": milestones}

    def get_event(self, event_ticker: str) -> Dict[str, object]:
        return self.get_json(f"events/{event_ticker}", {"with_nested_markets": "true"})

    def find_milestone_for_event(
        self,
        event_ticker: str,
        series_ticker: Optional[str],
    ) -> Dict[str, object]:
        payload = self.list_events_response(series_ticker=series_ticker, with_milestones=True)
        for milestone in payload["milestones"]:
            tickers = []
            related = milestone.get("related_event_tickers")
            primary = milestone.get("primary_event_tickers")
            if isinstance(related, list):
                tickers.extend(str(value) for value in related)
            if isinstance(primary, list):
                tickers.extend(str(value) for value in primary)
            details = milestone.get("details")
            main_ticker = details.get("main_game_event_ticker") if isinstance(details, dict) else None
            if event_ticker in tickers or main_ticker == event_ticker:
                return milestone
        raise ValueError(f"No milestone found for {event_ticker}")

    def list_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        tickers: Optional[str] = None,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        min_settled_ts: Optional[int] = None,
        max_settled_ts: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        return self.paginated(
            "markets",
            "markets",
            {
                "series_ticker": series_ticker,
                "event_ticker": event_ticker,
                "tickers": tickers,
                "status": status,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
                "min_settled_ts": min_settled_ts,
                "max_settled_ts": max_settled_ts,
                "limit": 1000,
            },
        )

    def get_market_candles(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int,
        include_latest_before_start: bool,
        historical_fallback: bool,
    ) -> List[Dict[str, object]]:
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
            "include_latest_before_start": bool_param(include_latest_before_start),
        }
        path = f"series/{series_ticker}/markets/{market_ticker}/candlesticks"
        try:
            payload = self.get_json(path, params)
        except requests.HTTPError:
            if not historical_fallback:
                raise
            payload = self.get_json(f"historical/markets/{market_ticker}/candlesticks", params)
        candles = payload.get("candlesticks", [])
        if not isinstance(candles, list):
            raise ValueError(f"Expected candlesticks list for {market_ticker}")
        return [candle for candle in candles if isinstance(candle, dict)]

    def get_live_data(self, milestone_id: str, include_player_stats: bool) -> Dict[str, object]:
        return self.get_json(
            f"live_data/milestone/{milestone_id}",
            {"include_player_stats": bool_param(include_player_stats)},
        )

    def get_game_stats(self, milestone_id: str) -> Dict[str, object]:
        return self.get_json(f"live_data/milestone/{milestone_id}/game_stats")


def clean_params(params: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in params.items() if value is not None and value != ""}


def bool_param(value: bool) -> str:
    return "true" if value else "false"


def parse_timestamp(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
        parsed = datetime.fromisoformat(stripped).replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    normalized = stripped.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def matches_query(row: Dict[str, object], query: Optional[str], fields: Sequence[str]) -> bool:
    if not query:
        return True
    haystack = " ".join(str(row.get(field) or "") for field in fields).lower()
    return all(term.lower() in haystack for term in query.split())


def write_rows(rows: List[Dict[str, object]], out_path: Optional[str]) -> None:
    if not rows:
        print("No rows.")
        return
    frame = pd.json_normalize(rows)
    if out_path:
        frame.to_csv(out_path, index=False)
        print(f"Wrote {len(frame)} rows to {out_path}")
        return
    print(frame.to_csv(index=False))


def write_json(payload: Dict[str, object], out_path: Optional[str]) -> None:
    text = json_dumps(payload)
    if out_path:
        Path(out_path).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote JSON to {out_path}")
        return
    print(text)


def json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, default=str)


def market_label(market: Dict[str, object]) -> str:
    return str(
        market.get("yes_sub_title")
        or market.get("title")
        or market.get("ticker")
        or ""
    )


def markets_from_event_payload(payload: Dict[str, object]) -> List[Dict[str, object]]:
    markets = payload.get("markets")
    if isinstance(markets, list) and markets:
        return [market for market in markets if isinstance(market, dict)]
    event = payload.get("event")
    if isinstance(event, dict):
        nested = event.get("markets")
        if isinstance(nested, list):
            return [market for market in nested if isinstance(market, dict)]
    return []


def series_from_event_payload(payload: Dict[str, object]) -> Optional[str]:
    event = payload.get("event")
    if isinstance(event, dict):
        series_ticker = event.get("series_ticker")
        if isinstance(series_ticker, str):
            return series_ticker
    return None


def decimal_or_nan(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def price_field(candle: Dict[str, object], group: str, field: str) -> float:
    nested = candle.get(group)
    if not isinstance(nested, dict):
        return math.nan
    return decimal_or_nan(nested.get(f"{field}_dollars", nested.get(field)))


def candles_to_dataframe(
    candles_by_market: Dict[str, List[Dict[str, object]]],
    markets_by_ticker: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ticker, candles in candles_by_market.items():
        market = markets_by_ticker.get(ticker, {})
        for candle in candles:
            end_period_ts = candle.get("end_period_ts")
            if not isinstance(end_period_ts, int):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "event_ticker": market.get("event_ticker"),
                    "market_title": market.get("title"),
                    "yes_sub_title": market.get("yes_sub_title"),
                    "result": market.get("result"),
                    "ts_utc": pd.to_datetime(end_period_ts, unit="s", utc=True),
                    "price_open": price_field(candle, "price", "open"),
                    "price_high": price_field(candle, "price", "high"),
                    "price_low": price_field(candle, "price", "low"),
                    "price_close": price_field(candle, "price", "close"),
                    "price_mean": price_field(candle, "price", "mean"),
                    "price_previous": price_field(candle, "price", "previous"),
                    "yes_bid_close": price_field(candle, "yes_bid", "close"),
                    "yes_ask_close": price_field(candle, "yes_ask", "close"),
                    "volume": decimal_or_nan(candle.get("volume_fp", candle.get("volume", 0))),
                    "open_interest": decimal_or_nan(candle.get("open_interest_fp", candle.get("open_interest", 0))),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["event_ticker", "ticker", "ts_utc"]).reset_index(drop=True)
    frame["implied_prob_pct"] = frame["price_close"] * 100
    return frame


def resolve_markets_for_candles(args: argparse.Namespace, client: KalshiClient) -> tuple[str, List[Dict[str, object]]]:
    series_ticker = args.series_ticker
    markets: List[Dict[str, object]] = []

    if args.event_ticker:
        event_payload = client.get_event(args.event_ticker)
        markets = markets_from_event_payload(event_payload)
        series_ticker = series_ticker or series_from_event_payload(event_payload)
        if not markets:
            markets = client.list_markets(event_ticker=args.event_ticker)

    if args.tickers:
        ticker_value = ",".join(split_csv_args(args.tickers))
        markets = client.list_markets(tickers=ticker_value)

    if args.query:
        markets = [
            market
            for market in markets
            if matches_query(market, args.query, ["ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title"])
        ]

    if not series_ticker:
        raise SystemExit("--series-ticker is required unless --event-ticker can supply it")
    if not markets:
        raise SystemExit("No markets matched the supplied candle options")
    return series_ticker, markets


def split_csv_args(values: Sequence[str]) -> List[str]:
    tickers: List[str] = []
    for value in values:
        tickers.extend(part.strip() for part in value.split(",") if part.strip())
    return tickers


def default_window(markets: Sequence[Dict[str, object]]) -> tuple[int, int]:
    open_times = [
        parse_iso_datetime(str(market["open_time"])).timestamp()
        for market in markets
        if market.get("open_time")
    ]
    close_times = [
        parse_iso_datetime(str(market["close_time"])).timestamp()
        for market in markets
        if market.get("close_time")
    ]
    if not open_times or not close_times:
        raise SystemExit("--start and --end are required when market open/close times are missing")
    return int(min(open_times)), int(max(close_times))


def plot_candles(frame: pd.DataFrame, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker, group in frame.groupby("ticker"):
        label_values = group["yes_sub_title"].dropna().unique()
        label = str(label_values[0]) if len(label_values) else str(ticker)
        ax.plot(group["ts_utc"], group["implied_prob_pct"], label=label, linewidth=1.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Yes implied probability (%)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved plot to {out_path}")


def flatten_play_by_play(stats_payload: Dict[str, object]) -> List[Dict[str, object]]:
    pbp = stats_payload.get("pbp")
    if not isinstance(pbp, dict):
        return []
    periods = pbp.get("periods")
    if not isinstance(periods, list):
        return []
    rows: List[Dict[str, object]] = []
    for period_index, period in enumerate(periods, start=1):
        if not isinstance(period, dict):
            continue
        events = period.get("events")
        if not isinstance(events, list):
            continue
        for event_index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue
            row = {
                "period_index": period_index,
                "event_index": event_index,
                "period_type": period.get("period_type") or period.get("type"),
                "period_number": period.get("period_number") or period.get("number") or period.get("period"),
            }
            row.update(event)
            rows.append(row)
    return rows


def resolve_milestone_id(args: argparse.Namespace, client: KalshiClient) -> str:
    if args.milestone_id:
        return args.milestone_id
    if not args.event_ticker:
        raise SystemExit("Provide --milestone-id or --event-ticker")
    milestone = client.find_milestone_for_event(args.event_ticker, args.series_ticker)
    milestone_id = milestone.get("id")
    if not isinstance(milestone_id, str):
        raise SystemExit(f"Milestone for {args.event_ticker} did not include an id")
    return milestone_id


def command_series(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    rows = client.list_series(args.category, args.tags, args.include_volume)
    rows = [
        row
        for row in rows
        if matches_query(row, args.search, ["ticker", "title", "category", "tags"])
    ]
    write_rows(rows, args.out)


def command_events(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    rows = client.list_events(
        series_ticker=args.series_ticker,
        status=args.status,
        min_close_ts=parse_timestamp(args.min_close),
        with_nested_markets=args.with_nested_markets,
    )
    rows = [
        row
        for row in rows
        if matches_query(row, args.query, ["event_ticker", "title", "sub_title", "category"])
    ]
    write_rows(rows, args.out)


def command_markets(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    rows = client.list_markets(
        series_ticker=args.series_ticker,
        event_ticker=args.event_ticker,
        tickers=",".join(split_csv_args(args.tickers)) if args.tickers else None,
        status=args.status,
        min_close_ts=parse_timestamp(args.min_close),
        max_close_ts=parse_timestamp(args.max_close),
        min_settled_ts=parse_timestamp(args.min_settled),
        max_settled_ts=parse_timestamp(args.max_settled),
    )
    rows = [
        row
        for row in rows
        if matches_query(row, args.query, ["ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title"])
    ]
    write_rows(rows, args.out)


def command_candles(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    series_ticker, markets = resolve_markets_for_candles(args, client)
    markets_by_ticker = {str(market["ticker"]): market for market in markets if market.get("ticker")}
    start_ts = parse_timestamp(args.start)
    end_ts = parse_timestamp(args.end)
    if start_ts is None or end_ts is None:
        default_start, default_end = default_window(markets)
        start_ts = start_ts or default_start
        end_ts = end_ts or default_end

    print(f"Pulling {len(markets_by_ticker)} market(s) from {format_utc(start_ts)} to {format_utc(end_ts)}")
    candles_by_market: Dict[str, List[Dict[str, object]]] = {}
    for ticker, market in markets_by_ticker.items():
        candles = client.get_market_candles(
            series_ticker=series_ticker,
            market_ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=args.interval,
            include_latest_before_start=args.include_latest_before_start,
            historical_fallback=not args.no_historical_fallback,
        )
        candles_by_market[ticker] = candles
        print(f"  {ticker:45s} {market_label(market):24s} {len(candles)} candles")

    frame = candles_to_dataframe(candles_by_market, markets_by_ticker)
    if frame.empty:
        raise SystemExit("No candle rows returned")
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} rows to {args.out}")

    if args.metadata_out:
        pd.json_normalize(markets).to_csv(args.metadata_out, index=False)
        print(f"Wrote market metadata to {args.metadata_out}")
    if args.plot:
        plot_candles(frame, args.plot)


def command_milestones(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    if args.event_ticker:
        milestone = client.find_milestone_for_event(args.event_ticker, args.series_ticker)
        rows = [milestone]
    else:
        payload = client.list_events_response(
            series_ticker=args.series_ticker,
            status=args.status,
            with_milestones=True,
        )
        rows = payload["milestones"]
    rows = [
        row
        for row in rows
        if matches_query(row, args.query, ["id", "title", "type", "related_event_tickers", "primary_event_tickers"])
    ]
    write_rows(rows, args.out)


def command_live_data(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    milestone_id = resolve_milestone_id(args, client)
    payload = client.get_live_data(milestone_id, args.include_player_stats)
    write_json(payload, args.out)


def command_game_stats(args: argparse.Namespace) -> None:
    client = KalshiClient(args.base_url)
    milestone_id = resolve_milestone_id(args, client)
    payload = client.get_game_stats(milestone_id)
    if args.json_out:
        write_json(payload, args.json_out)
    rows = flatten_play_by_play(payload)
    if args.events_out:
        write_rows(rows, args.events_out)
    elif not args.json_out:
        write_rows(rows, None)
    print(f"Found {len(rows)} play-by-play events")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull public Kalshi market data")
    parser.add_argument("--base-url", default=BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    series = subparsers.add_parser("series", help="List series")
    series.add_argument("--category")
    series.add_argument("--tags")
    series.add_argument("--search")
    series.add_argument("--include-volume", action="store_true")
    series.add_argument("--out")
    series.set_defaults(func=command_series)

    events = subparsers.add_parser("events", help="List events")
    events.add_argument("--series-ticker")
    events.add_argument("--status", choices=["unopened", "open", "closed", "settled"])
    events.add_argument("--min-close")
    events.add_argument("--query")
    events.add_argument("--with-nested-markets", action="store_true")
    events.add_argument("--out")
    events.set_defaults(func=command_events)

    markets = subparsers.add_parser("markets", help="List markets")
    markets.add_argument("--series-ticker")
    markets.add_argument("--event-ticker")
    markets.add_argument("--tickers", action="append")
    markets.add_argument("--status", choices=["unopened", "open", "paused", "closed", "settled"])
    markets.add_argument("--min-close")
    markets.add_argument("--max-close")
    markets.add_argument("--min-settled")
    markets.add_argument("--max-settled")
    markets.add_argument("--query")
    markets.add_argument("--out")
    markets.set_defaults(func=command_markets)

    candles = subparsers.add_parser("candles", help="Pull candlesticks")
    candles.add_argument("--series-ticker")
    candles.add_argument("--event-ticker")
    candles.add_argument("--tickers", action="append")
    candles.add_argument("--query")
    candles.add_argument("--start")
    candles.add_argument("--end")
    candles.add_argument("--interval", type=int, default=1, choices=[1, 60, 1440])
    candles.add_argument("--include-latest-before-start", action="store_true")
    candles.add_argument("--no-historical-fallback", action="store_true")
    candles.add_argument("--out", required=True)
    candles.add_argument("--metadata-out")
    candles.add_argument("--plot")
    candles.set_defaults(func=command_candles)

    milestones = subparsers.add_parser("milestones", help="List or find event milestones")
    milestones.add_argument("--series-ticker")
    milestones.add_argument("--event-ticker")
    milestones.add_argument("--status", choices=["unopened", "open", "closed", "settled"])
    milestones.add_argument("--query")
    milestones.add_argument("--out")
    milestones.set_defaults(func=command_milestones)

    live_data = subparsers.add_parser("live-data", help="Pull live data for a milestone")
    live_data.add_argument("--milestone-id")
    live_data.add_argument("--event-ticker")
    live_data.add_argument("--series-ticker", default="KXNBAGAME")
    live_data.add_argument("--include-player-stats", action="store_true")
    live_data.add_argument("--out")
    live_data.set_defaults(func=command_live_data)

    game_stats = subparsers.add_parser("game-stats", help="Pull game stats/play-by-play for a milestone")
    game_stats.add_argument("--milestone-id")
    game_stats.add_argument("--event-ticker")
    game_stats.add_argument("--series-ticker", default="KXNBAGAME")
    game_stats.add_argument("--json-out")
    game_stats.add_argument("--events-out")
    game_stats.set_defaults(func=command_game_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
