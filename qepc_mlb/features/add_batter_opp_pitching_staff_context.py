#!/usr/bin/env python3
"""
Add opponent pitching staff context to batter-game rows.

Input:
  Batter logs, usually:
    cache/mlb/processed/batter_game_logs_opp_pitcher_matchup.parquet

  Pitcher logs, usually:
    cache/mlb/processed/pitcher_game_logs.parquet

Output:
  Batter logs with opponent staff prior-form features.

Safety:
  All opponent staff features are calculated from PRIOR games only.
  No same-game pitching outcomes are used for the current batter row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SAFETY_VERSION = "batter_opp_pitching_staff_context_v1_pregame_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batters", required=True)
    p.add_argument("--pitchers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    elif "date" in df.columns:
        raw = df["date"].astype("string").str.replace(r"\.0$", "", regex=True)
        df["game_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("Missing game_date/date column.")

    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def coalesce_numeric(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype="float64")

    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            out = out.where(out.notna(), s)

    return out


def coerce_boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(False, index=s.index, dtype=bool)
    out.loc[numeric.fillna(0) != 0] = True

    text = s.astype("string").str.strip().str.lower()
    out.loc[text.isin({"true", "t", "yes", "y", "starter", "start", "starting", "1", "1.0"})] = True
    out.loc[text.isin({"false", "f", "no", "n", "0", "0.0", "nan", "<na>", "none", ""})] = False

    return out


def normalize_pitcher_logs(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p = ensure_game_date(p)

    # Retrosheet-style pitching files sometimes include value + official duplicated stat rows.
    if "stattype" in p.columns:
        p = p[p["stattype"].astype("string").str.lower().eq("value")].copy()

    game_col = first_existing(p, ["game_id", "gid"])
    team_col = first_existing(p, ["team", "pitcher_team"])
    pitcher_col = first_existing(p, ["pitcher_id", "id", "player_id"])

    if game_col is None:
        raise ValueError("Pitcher logs missing game_id/gid column.")
    if team_col is None:
        raise ValueError("Pitcher logs missing team/pitcher_team column.")

    out = pd.DataFrame(index=p.index)
    out["game_id"] = p[game_col].astype("string")
    out["pitcher_team"] = p[team_col].astype("string")
    out["game_date"] = p["game_date"]
    out["season"] = p["season"]

    if pitcher_col is not None:
        out["pitcher_id"] = p[pitcher_col].astype("string")
    else:
        out["pitcher_id"] = pd.NA

    if "is_starter" in p.columns:
        out["is_starter"] = coerce_boolish(p["is_starter"])
    elif "p_gs" in p.columns:
        out["is_starter"] = pd.to_numeric(p["p_gs"], errors="coerce").fillna(0).astype(float).ne(0)
    else:
        out["is_starter"] = False

    out["strikeouts"] = coalesce_numeric(p, ["strikeouts", "p_k"])
    out["batters_faced"] = coalesce_numeric(p, ["batters_faced", "p_bfp"])
    out["hits_allowed"] = coalesce_numeric(p, ["hits_allowed", "p_h"])
    out["walks_allowed"] = coalesce_numeric(p, ["walks_allowed", "walks", "p_w"])
    out["runs_allowed"] = coalesce_numeric(p, ["runs_allowed", "p_r"])
    out["earned_runs"] = coalesce_numeric(p, ["earned_runs", "p_er"])
    out["home_runs_allowed"] = coalesce_numeric(p, ["home_runs_allowed", "p_hr"])
    out["outs_recorded"] = coalesce_numeric(p, ["outs_recorded", "p_ipouts"])

    if "innings_pitched" in p.columns:
        out["innings_pitched"] = pd.to_numeric(p["innings_pitched"], errors="coerce")
    else:
        out["innings_pitched"] = out["outs_recorded"] / 3.0

    out = out.dropna(subset=["game_id", "pitcher_team", "game_date"]).copy()

    stat_cols = [
        "strikeouts",
        "batters_faced",
        "hits_allowed",
        "walks_allowed",
        "runs_allowed",
        "earned_runs",
        "home_runs_allowed",
        "outs_recorded",
        "innings_pitched",
    ]

    for c in stat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def aggregate_team_pitching_by_game(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()

    stat_cols = [
        "strikeouts",
        "batters_faced",
        "hits_allowed",
        "walks_allowed",
        "runs_allowed",
        "earned_runs",
        "home_runs_allowed",
        "outs_recorded",
        "innings_pitched",
    ]

    for c in stat_cols:
        p[f"bullpen_{c}"] = np.where(~p["is_starter"].fillna(False).astype(bool), p[c], 0.0)

    p["pitcher_appearance_count"] = 1
    p["bullpen_pitcher_count"] = np.where(~p["is_starter"].fillna(False).astype(bool), 1, 0)

    agg_cols = {c: "sum" for c in stat_cols}
    agg_cols.update({f"bullpen_{c}": "sum" for c in stat_cols})
    agg_cols.update(
        {
            "pitcher_appearance_count": "sum",
            "bullpen_pitcher_count": "sum",
        }
    )

    team_game = (
        p.groupby(["game_id", "pitcher_team", "game_date", "season"], dropna=False)
        .agg(agg_cols)
        .reset_index()
    )

    team_game["bullpen_used"] = team_game["bullpen_pitcher_count"].gt(0).astype(int)

    return team_game


def rolling_rate(num: pd.Series, den: pd.Series, window: int) -> pd.Series:
    num_s = pd.to_numeric(num, errors="coerce").fillna(0).shift(1)
    den_s = pd.to_numeric(den, errors="coerce").fillna(0).shift(1)
    den_roll = den_s.rolling(window, min_periods=1).sum().replace(0, np.nan)
    return num_s.rolling(window, min_periods=1).sum() / den_roll


def add_team_prior_features(team_game: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    pieces = []

    groups = team_game.groupby("pitcher_team", sort=False)
    iterator = groups

    if show_progress:
        iterator = tqdm(groups, total=team_game["pitcher_team"].nunique(), desc="Building team pitching prior features")

    for _, g in iterator:
        g = g.sort_values(["game_date", "game_id"]).copy()
        g["opp_pitching_staff_games_prior"] = np.arange(len(g), dtype=float)

        basic_stats = [
            "batters_faced",
            "strikeouts",
            "hits_allowed",
            "walks_allowed",
            "runs_allowed",
            "earned_runs",
            "home_runs_allowed",
            "outs_recorded",
            "innings_pitched",
            "bullpen_batters_faced",
            "bullpen_strikeouts",
            "bullpen_hits_allowed",
            "bullpen_walks_allowed",
            "bullpen_runs_allowed",
            "bullpen_earned_runs",
            "bullpen_home_runs_allowed",
            "bullpen_outs_recorded",
            "bullpen_innings_pitched",
            "pitcher_appearance_count",
            "bullpen_pitcher_count",
            "bullpen_used",
        ]

        for stat in basic_stats:
            if stat not in g.columns:
                continue

            shifted = pd.to_numeric(g[stat], errors="coerce").shift(1)
            g[f"opp_pitching_staff_{stat}_lag1"] = shifted

            for w in [3, 5, 10, 20]:
                g[f"opp_pitching_staff_{stat}_roll{w}"] = shifted.rolling(w, min_periods=1).mean()

        rate_specs = {
            "k_rate_allowed": ("strikeouts", "batters_faced"),
            "hit_rate_allowed": ("hits_allowed", "batters_faced"),
            "bb_rate_allowed": ("walks_allowed", "batters_faced"),
            "hr_rate_allowed": ("home_runs_allowed", "batters_faced"),
            "run_rate_allowed": ("runs_allowed", "batters_faced"),
            "er_per_ip": ("earned_runs", "innings_pitched"),

            "bullpen_k_rate_allowed": ("bullpen_strikeouts", "bullpen_batters_faced"),
            "bullpen_hit_rate_allowed": ("bullpen_hits_allowed", "bullpen_batters_faced"),
            "bullpen_bb_rate_allowed": ("bullpen_walks_allowed", "bullpen_batters_faced"),
            "bullpen_hr_rate_allowed": ("bullpen_home_runs_allowed", "bullpen_batters_faced"),
            "bullpen_run_rate_allowed": ("bullpen_runs_allowed", "bullpen_batters_faced"),
            "bullpen_er_per_ip": ("bullpen_earned_runs", "bullpen_innings_pitched"),

            "bullpen_out_share": ("bullpen_outs_recorded", "outs_recorded"),
            "bullpen_bf_share": ("bullpen_batters_faced", "batters_faced"),
        }

        for rate_name, (num_col, den_col) in rate_specs.items():
            if num_col not in g.columns or den_col not in g.columns:
                continue

            num_shift = pd.to_numeric(g[num_col], errors="coerce").shift(1)
            den_shift = pd.to_numeric(g[den_col], errors="coerce").shift(1).replace(0, np.nan)

            g[f"opp_pitching_staff_{rate_name}_lag1"] = num_shift / den_shift

            for w in [3, 5, 10, 20]:
                g[f"opp_pitching_staff_{rate_name}_roll{w}"] = rolling_rate(g[num_col], g[den_col], w)

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return out


def main() -> None:
    args = parse_args()

    bat = read_any(args.batters).copy()
    pit = normalize_pitcher_logs(read_any(args.pitchers))

    bat = ensure_game_date(bat)

    # Remove old staff context columns if rerunning.
    old_staff_cols = [
        c for c in bat.columns
        if c.startswith("opp_pitching_staff_") or c == "has_opp_pitching_staff_context"
    ]
    if old_staff_cols:
        bat = bat.drop(columns=old_staff_cols)

    team_game = aggregate_team_pitching_by_game(pit)
    team_context = add_team_prior_features(team_game, show_progress=not args.no_progress)

    context_cols = [
        "game_id",
        "pitcher_team",
    ] + [c for c in team_context.columns if c.startswith("opp_pitching_staff_")]

    ctx = team_context[context_cols].copy()
    ctx["_game_key"] = ctx["game_id"].astype("string")
    ctx["_opp_team_key"] = ctx["pitcher_team"].astype("string")

    ctx = ctx.drop(columns=["game_id", "pitcher_team"])
    ctx = ctx.drop_duplicates(["_game_key", "_opp_team_key"], keep="first").copy()

    game_col = first_existing(bat, ["game_id", "gid"])
    opp_col = first_existing(bat, ["opponent", "opp"])

    if game_col is None:
        raise ValueError("Batter logs missing game_id/gid column.")
    if opp_col is None:
        raise ValueError("Batter logs missing opponent/opp column.")

    bat["_game_key"] = bat[game_col].astype("string")
    bat["_opp_team_key"] = bat[opp_col].astype("string")

    before_rows = len(bat)
    before_cols = set(bat.columns)

    merged = bat.merge(ctx, how="left", on=["_game_key", "_opp_team_key"])
    merged["has_opp_pitching_staff_context"] = merged["opp_pitching_staff_games_prior"].notna()

    merged = merged.drop(columns=["_game_key", "_opp_team_key"], errors="ignore")

    duplicate_cols = merged.columns[merged.columns.duplicated()].tolist()
    if duplicate_cols:
        print("Dropping duplicate columns before save:", duplicate_cols)
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    if len(merged) != before_rows:
        raise ValueError(f"Row count changed after merge: before={before_rows}, after={len(merged)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        merged.to_csv(csv_path, index=False)

    added_cols = [c for c in merged.columns if c not in before_cols]
    staff_feature_cols = [
        c for c in added_cols
        if c.startswith("opp_pitching_staff_") or c == "has_opp_pitching_staff_context"
    ]

    summary = {
        "safety_version": SAFETY_VERSION,
        "batters_input": args.batters,
        "pitchers_input": args.pitchers,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(merged)),
        "date_min": str(merged["game_date"].min().date()) if merged["game_date"].notna().any() else None,
        "date_max": str(merged["game_date"].max().date()) if merged["game_date"].notna().any() else None,
        "seasons": [int(x) for x in sorted(merged["season"].dropna().astype(int).unique().tolist())],
        "pitcher_rows_used": int(len(pit)),
        "pitching_team_game_rows": int(len(team_game)),
        "bullpen_pitcher_rows_used": int((~pit["is_starter"].fillna(False).astype(bool)).sum()),
        "staff_context_match_count": int(merged["has_opp_pitching_staff_context"].sum()),
        "staff_context_match_rate": float(merged["has_opp_pitching_staff_context"].mean()),
        "added_column_count": int(len(added_cols)),
        "staff_feature_column_count": int(len(staff_feature_cols)),
        "staff_feature_columns": staff_feature_cols,
        "pregame_safety_note": "All opp_pitching_staff_* features are shifted/rolling prior-game values only.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
