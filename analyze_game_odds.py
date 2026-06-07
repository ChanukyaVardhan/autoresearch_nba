#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one Kalshi NBA game odds file.")
    parser.add_argument("--candles", required=True, help="Long-form candle CSV from kalshi_pull.py")
    parser.add_argument("--markets", required=True, help="Market metadata CSV from kalshi_pull.py")
    parser.add_argument("--pbp", help="Optional play-by-play CSV from kalshi_pull.py game-stats")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix without extension")
    return parser.parse_args()


def infer_teams(title: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(r"(?:Game\s+\d+:\s*)?(?P<away>.+?) at (?P<home>.+?) Winner\?", title)
    if match is None:
        return None, None
    return match.group("away"), match.group("home")


def first_crossing_time(frame: pd.DataFrame, column: str, threshold: float) -> Optional[pd.Timestamp]:
    rows = frame.loc[frame[column] >= threshold, "ts_utc"]
    if rows.empty:
        return None
    return rows.iloc[0]


def largest_move(frame: pd.DataFrame, team_columns: list[str], window: int) -> dict[str, object]:
    best_team = ""
    best_ts = pd.NaT
    best_move = 0.0
    best_from = float("nan")
    best_to = float("nan")

    for team in team_columns:
        deltas = frame[team].diff(window)
        absolute = deltas.abs()
        if absolute.dropna().empty:
            continue
        idx = absolute.idxmax()
        move = deltas.loc[idx]
        if pd.isna(move):
            continue
        if abs(float(move)) > abs(best_move):
            best_team = team
            best_ts = frame.loc[idx, "ts_utc"]
            best_move = float(move)
            best_from = float(frame.loc[idx - window, team]) if idx >= window else float("nan")
            best_to = float(frame.loc[idx, team])

    return {
        "team": best_team,
        "ts_utc": best_ts,
        "move_pct": best_move,
        "from_prob_pct": best_from,
        "to_prob_pct": best_to,
    }


def load_probabilities(candles_path: Path) -> pd.DataFrame:
    candles = pd.read_csv(candles_path, parse_dates=["ts_utc"])
    if candles.empty:
        raise ValueError(f"No candle rows in {candles_path}")

    probabilities = (
        candles.pivot_table(
            index="ts_utc",
            columns="yes_sub_title",
            values="implied_prob_pct",
            aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    probabilities.columns.name = None

    team_columns = [column for column in probabilities.columns if column != "ts_utc"]
    probabilities = probabilities.dropna(subset=team_columns).reset_index(drop=True)
    probabilities["prob_sum_pct"] = probabilities[team_columns].sum(axis=1)
    probabilities["market_leader"] = probabilities[team_columns].idxmax(axis=1)
    probabilities["market_leader_prob_pct"] = probabilities[team_columns].max(axis=1)
    probabilities["market_spread_pct"] = probabilities[team_columns].max(axis=1) - probabilities[team_columns].min(axis=1)
    return probabilities


def load_scoreboard(pbp_path: Optional[Path], home_team: Optional[str], away_team: Optional[str]) -> Optional[pd.DataFrame]:
    if pbp_path is None or not pbp_path.exists():
        return None

    pbp = pd.read_csv(pbp_path)
    if pbp.empty or "wall_clock" not in pbp.columns:
        return None

    pbp = pbp.dropna(subset=["wall_clock"]).copy()
    pbp["event_ts_utc"] = pd.to_datetime(pbp["wall_clock"], unit="s", utc=True)
    pbp = pbp.sort_values("event_ts_utc")
    score = pbp[["event_ts_utc", "period_number", "clock", "away_points", "home_points", "description"]].copy()
    score["score_margin_home_minus_away"] = score["home_points"] - score["away_points"]
    if home_team:
        score["home_team"] = home_team
    if away_team:
        score["away_team"] = away_team
    return score


def load_final_score(pbp_path: Optional[Path]) -> Optional[tuple[int, int]]:
    if pbp_path is None or not pbp_path.exists():
        return None
    pbp = pd.read_csv(pbp_path)
    if pbp.empty or "away_points" not in pbp.columns or "home_points" not in pbp.columns:
        return None
    final = pbp.dropna(subset=["away_points", "home_points"])
    if final.empty:
        return None
    row = final.iloc[0]
    return int(row["away_points"]), int(row["home_points"])


def align_score(probabilities: pd.DataFrame, score: Optional[pd.DataFrame]) -> pd.DataFrame:
    if score is None:
        return probabilities
    return pd.merge_asof(
        probabilities.sort_values("ts_utc"),
        score.sort_values("event_ts_utc"),
        left_on="ts_utc",
        right_on="event_ts_utc",
        direction="backward",
    )


def minutes_after(start: Optional[pd.Timestamp], value: Optional[pd.Timestamp]) -> Optional[float]:
    if start is None or value is None or pd.isna(start) or pd.isna(value):
        return None
    return round((value - start).total_seconds() / 60, 2)


def write_summary(
    probabilities: pd.DataFrame,
    markets: pd.DataFrame,
    score: Optional[pd.DataFrame],
    final_score: Optional[tuple[int, int]],
    out_path: Path,
) -> dict[str, object]:
    title = str(markets["title"].dropna().iloc[0])
    event_ticker = str(markets["event_ticker"].dropna().iloc[0])
    away_team, home_team = infer_teams(title)
    winner_rows = markets.loc[markets["result"] == "yes"]
    winner_team = str(winner_rows["yes_sub_title"].iloc[0]) if not winner_rows.empty else ""
    team_columns = [column for column in probabilities.columns if column not in {"ts_utc", "prob_sum_pct", "market_leader", "market_leader_prob_pct", "market_spread_pct"}]

    game_start = None
    game_end = None
    final_home_points = None
    final_away_points = None
    if score is not None and not score.empty:
        game_start = score["event_ts_utc"].min()
        game_end = score["event_ts_utc"].max()
        final = score.sort_values("event_ts_utc").iloc[-1]
        final_home_points = int(final["home_points"])
        final_away_points = int(final["away_points"])
    elif final_score is not None:
        final_away_points, final_home_points = final_score

    one_min = largest_move(probabilities, team_columns, 1)
    five_min = largest_move(probabilities, team_columns, 5)

    leader_changes = int((probabilities["market_leader"] != probabilities["market_leader"].shift()).sum() - 1)
    winner_cross_70 = first_crossing_time(probabilities, winner_team, 70.0) if winner_team in probabilities else None
    winner_cross_90 = first_crossing_time(probabilities, winner_team, 90.0) if winner_team in probabilities else None

    summary = {
        "event_ticker": event_ticker,
        "title": title,
        "away_team": away_team,
        "home_team": home_team,
        "winner_team": winner_team,
        "final_score": f"{away_team} {final_away_points}, {home_team} {final_home_points}" if final_home_points is not None and final_away_points is not None else "",
        "window_start_utc": probabilities["ts_utc"].min(),
        "window_end_utc": probabilities["ts_utc"].max(),
        "game_start_utc_from_pbp": game_start,
        "game_end_utc_from_pbp": game_end,
        "winner_prob_window_start_pct": round(float(probabilities[winner_team].dropna().iloc[0]), 2) if winner_team in probabilities else None,
        "winner_prob_game_start_pct": round(float(probabilities.loc[probabilities["ts_utc"] >= game_start, winner_team].dropna().iloc[0]), 2) if winner_team in probabilities and game_start is not None else None,
        "winner_prob_window_end_pct": round(float(probabilities[winner_team].dropna().iloc[-1]), 2) if winner_team in probabilities else None,
        "winner_min_prob_pct": round(float(probabilities[winner_team].min()), 2) if winner_team in probabilities else None,
        "winner_max_prob_pct": round(float(probabilities[winner_team].max()), 2) if winner_team in probabilities else None,
        "winner_cross_70_utc": winner_cross_70,
        "winner_cross_70_minutes_after_game_start": minutes_after(game_start, winner_cross_70),
        "winner_cross_90_utc": winner_cross_90,
        "winner_cross_90_minutes_after_game_start": minutes_after(game_start, winner_cross_90),
        "leader_changes": leader_changes,
        "largest_1min_move_team": one_min["team"],
        "largest_1min_move_utc": one_min["ts_utc"],
        "largest_1min_move_pct": round(float(one_min["move_pct"]), 2),
        "largest_5min_move_team": five_min["team"],
        "largest_5min_move_utc": five_min["ts_utc"],
        "largest_5min_move_pct": round(float(five_min["move_pct"]), 2),
    }
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return summary


def write_top_moves(aligned: pd.DataFrame, out_path: Path) -> None:
    excluded_columns = {
        "ts_utc",
        "prob_sum_pct",
        "market_leader",
        "market_leader_prob_pct",
        "market_spread_pct",
        "event_ts_utc",
        "period_number",
        "clock",
        "away_points",
        "home_points",
        "description",
        "score_margin_home_minus_away",
        "home_team",
        "away_team",
    }
    team_columns = [column for column in aligned.columns if column not in excluded_columns]
    rows: list[dict[str, object]] = []

    for window in [1, 5]:
        for team in team_columns:
            delta_column = aligned[team].diff(window)
            for idx in delta_column.abs().nlargest(10).index:
                if pd.isna(delta_column.loc[idx]):
                    continue
                rows.append(
                    {
                        "window_minutes": window,
                        "team": team,
                        "ts_utc": aligned.loc[idx, "ts_utc"],
                        "move_pct": round(float(delta_column.loc[idx]), 2),
                        "from_prob_pct": round(float(aligned.loc[idx - window, team]), 2) if idx >= window else None,
                        "to_prob_pct": round(float(aligned.loc[idx, team]), 2),
                        "period_number": aligned.loc[idx, "period_number"] if "period_number" in aligned else None,
                        "clock": aligned.loc[idx, "clock"] if "clock" in aligned else None,
                        "away_points": aligned.loc[idx, "away_points"] if "away_points" in aligned else None,
                        "home_points": aligned.loc[idx, "home_points"] if "home_points" in aligned else None,
                        "score_margin_home_minus_away": aligned.loc[idx, "score_margin_home_minus_away"] if "score_margin_home_minus_away" in aligned else None,
                        "description": aligned.loc[idx, "description"] if "description" in aligned else None,
                    }
                )

    top_moves = pd.DataFrame(rows)
    top_moves["abs_move_pct"] = top_moves["move_pct"].abs()
    top_moves = top_moves.sort_values(["abs_move_pct", "window_minutes"], ascending=[False, True])
    top_moves.to_csv(out_path, index=False)


def plot(probabilities: pd.DataFrame, score: Optional[pd.DataFrame], summary: dict[str, object], out_path: Path) -> None:
    team_columns = [
        column
        for column in probabilities.columns
        if column not in {"ts_utc", "prob_sum_pct", "market_leader", "market_leader_prob_pct", "market_spread_pct"}
    ]
    has_score = score is not None and not score.empty
    fig, axes = plt.subplots(
        2 if has_score else 1,
        1,
        figsize=(14, 8 if has_score else 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]} if has_score else None,
    )
    odds_axis = axes[0] if has_score else axes

    for team in team_columns:
        odds_axis.plot(probabilities["ts_utc"], probabilities[team], linewidth=1.8, label=team)
    odds_axis.set_ylim(0, 100)
    odds_axis.set_ylabel("Kalshi implied win probability (%)")
    odds_axis.set_title(str(summary["title"]))
    odds_axis.grid(True, alpha=0.25)
    odds_axis.legend(loc="upper left")

    game_start = summary.get("game_start_utc_from_pbp")
    game_end = summary.get("game_end_utc_from_pbp")
    for value, label in [(game_start, "tipoff"), (game_end, "final")]:
        if value is not None and not pd.isna(value):
            odds_axis.axvline(value, color="#444444", linestyle="--", linewidth=1.0, alpha=0.65)
            odds_axis.text(value, 4, label, rotation=90, va="bottom", ha="right", color="#444444")

    if has_score:
        score_axis = axes[1]
        score_axis.step(score["event_ts_utc"], score["score_margin_home_minus_away"], where="post", color="#333333", linewidth=1.4)
        score_axis.axhline(0, color="#999999", linewidth=0.8)
        score_axis.set_ylabel("Home margin")
        score_axis.set_xlabel("Time (UTC)")
        score_axis.grid(True, alpha=0.2)
    else:
        odds_axis.set_xlabel("Time (UTC)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main_from_paths(candles_path: Path, markets_path: Path, pbp_path: Optional[Path], out_prefix: Path) -> None:
    markets = pd.read_csv(markets_path)
    title = str(markets["title"].dropna().iloc[0])
    away_team, home_team = infer_teams(title)

    probabilities = load_probabilities(candles_path)
    score = load_scoreboard(pbp_path, home_team, away_team)
    final_score = load_final_score(pbp_path)
    aligned = align_score(probabilities, score)

    aligned.to_csv(out_prefix.with_name(f"{out_prefix.name}_probabilities_wide.csv"), index=False)
    write_top_moves(aligned, out_prefix.with_name(f"{out_prefix.name}_top_odds_moves.csv"))
    summary = write_summary(
        probabilities,
        markets,
        score,
        final_score,
        out_prefix.with_name(f"{out_prefix.name}_signal_summary.csv"),
    )
    plot(probabilities, score, summary, out_prefix.with_name(f"{out_prefix.name}_odds_score_chart.png"))


def main() -> None:
    args = parse_args()
    main_from_paths(
        candles_path=Path(args.candles),
        markets_path=Path(args.markets),
        pbp_path=Path(args.pbp) if args.pbp else None,
        out_prefix=Path(args.out_prefix),
    )


if __name__ == "__main__":
    main()
