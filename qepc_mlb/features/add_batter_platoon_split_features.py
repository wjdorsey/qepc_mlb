#!/usr/bin/env python3
"""
Add leakage-safe batter platoon split history features.

Input:
  cache/mlb/processed/batter_game_logs_opp_pitcher_matchup.parquet

Output:
  cache/mlb/processed/batter_game_logs_opp_pitcher_matchup_platoon.parquet

Safety:
  Every platoon feature is calculated using only games BEFORE the current row.
  No same-game batter outcome is used as a model feature for that same game.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SAFETY_VERSION = "batter_platoon_split_features_v1_pregame_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
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


def normalize_hand(x: Any) -> str:
    if pd.isna(x):
        return "U"

    s = str(x).strip().upper()

    if s in {"R", "RIGHT", "RIGHTY"}:
        return "R"
    if s in {"L", "LEFT", "LEFTY"}:
        return "L"
    if s in {"S", "B", "SW", "SWITCH", "BOTH"}:
        return "S"

    return "U"


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def prior_cumsum_by_condition(g: pd.DataFrame, stat_col: str, condition: pd.Series) -> pd.Series:
    values = pd.to_numeric(g[stat_col], errors="coerce").fillna(0)
    masked = values.where(condition, 0)
    return masked.cumsum() - masked


def season_prior_cumsum_by_condition(g: pd.DataFrame, stat_col: str, condition: pd.Series) -> pd.Series:
    values = pd.to_numeric(g[stat_col], errors="coerce").fillna(0)
    masked = values.where(condition, 0)
    return masked.groupby(g["season"]).cumsum() - masked


def add_group_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["game_date", "game_id" if "game_id" in g.columns else "batter_id"]).copy()

    # Required stats, defaulting safely to zero if a column is absent.
    for c in ["hit_1plus", "hits", "at_bats", "plate_appearances", "walks", "strikeouts", "total_bases"]:
        if c not in g.columns:
            g[c] = 0

    if "opp_starter_throw_norm" not in g.columns:
        if "opp_starter_throw" in g.columns:
            g["opp_starter_throw_norm"] = g["opp_starter_throw"].map(normalize_hand)
        else:
            g["opp_starter_throw_norm"] = "U"

    if "season" not in g.columns:
        g["season"] = pd.to_datetime(g["game_date"], errors="coerce").dt.year

    for hand, tag in [("R", "rhp"), ("L", "lhp")]:
        cond = g["opp_starter_throw_norm"].eq(hand)

        games_prior = cond.astype(int).cumsum() - cond.astype(int)
        games_season_prior = cond.astype(int).groupby(g["season"]).cumsum() - cond.astype(int)

        h1_prior = prior_cumsum_by_condition(g, "hit_1plus", cond)
        hits_prior = prior_cumsum_by_condition(g, "hits", cond)
        ab_prior = prior_cumsum_by_condition(g, "at_bats", cond)
        pa_prior = prior_cumsum_by_condition(g, "plate_appearances", cond)
        walks_prior = prior_cumsum_by_condition(g, "walks", cond)
        ks_prior = prior_cumsum_by_condition(g, "strikeouts", cond)
        tb_prior = prior_cumsum_by_condition(g, "total_bases", cond)

        h1_season_prior = season_prior_cumsum_by_condition(g, "hit_1plus", cond)
        hits_season_prior = season_prior_cumsum_by_condition(g, "hits", cond)
        ab_season_prior = season_prior_cumsum_by_condition(g, "at_bats", cond)
        pa_season_prior = season_prior_cumsum_by_condition(g, "plate_appearances", cond)
        walks_season_prior = season_prior_cumsum_by_condition(g, "walks", cond)
        ks_season_prior = season_prior_cumsum_by_condition(g, "strikeouts", cond)
        tb_season_prior = season_prior_cumsum_by_condition(g, "total_bases", cond)

        # Career-to-date split before current game.
        g[f"batter_games_prior_vs_{tag}"] = games_prior.astype(float)
        g[f"batter_hit_1plus_rate_prior_vs_{tag}"] = safe_div(h1_prior, games_prior)
        g[f"batter_ba_prior_vs_{tag}"] = safe_div(hits_prior, ab_prior)
        g[f"batter_k_rate_prior_vs_{tag}"] = safe_div(ks_prior, pa_prior)
        g[f"batter_bb_rate_prior_vs_{tag}"] = safe_div(walks_prior, pa_prior)
        g[f"batter_tb_per_ab_prior_vs_{tag}"] = safe_div(tb_prior, ab_prior)

        # Same-season-to-date split before current game.
        g[f"batter_games_season_prior_vs_{tag}"] = games_season_prior.astype(float)
        g[f"batter_hit_1plus_rate_season_prior_vs_{tag}"] = safe_div(h1_season_prior, games_season_prior)
        g[f"batter_ba_season_prior_vs_{tag}"] = safe_div(hits_season_prior, ab_season_prior)
        g[f"batter_k_rate_season_prior_vs_{tag}"] = safe_div(ks_season_prior, pa_season_prior)
        g[f"batter_bb_rate_season_prior_vs_{tag}"] = safe_div(walks_season_prior, pa_season_prior)
        g[f"batter_tb_per_ab_season_prior_vs_{tag}"] = safe_div(tb_season_prior, ab_season_prior)

    # Current matchup-selected platoon features.
    is_rhp = g["opp_starter_throw_norm"].eq("R")
    is_lhp = g["opp_starter_throw_norm"].eq("L")

    pairs = [
        ("batter_games_prior_vs_pitcher_hand", "batter_games_prior_vs_rhp", "batter_games_prior_vs_lhp"),
        ("batter_hit_1plus_rate_prior_vs_pitcher_hand", "batter_hit_1plus_rate_prior_vs_rhp", "batter_hit_1plus_rate_prior_vs_lhp"),
        ("batter_ba_prior_vs_pitcher_hand", "batter_ba_prior_vs_rhp", "batter_ba_prior_vs_lhp"),
        ("batter_k_rate_prior_vs_pitcher_hand", "batter_k_rate_prior_vs_rhp", "batter_k_rate_prior_vs_lhp"),
        ("batter_bb_rate_prior_vs_pitcher_hand", "batter_bb_rate_prior_vs_rhp", "batter_bb_rate_prior_vs_lhp"),
        ("batter_tb_per_ab_prior_vs_pitcher_hand", "batter_tb_per_ab_prior_vs_rhp", "batter_tb_per_ab_prior_vs_lhp"),

        ("batter_games_season_prior_vs_pitcher_hand", "batter_games_season_prior_vs_rhp", "batter_games_season_prior_vs_lhp"),
        ("batter_hit_1plus_rate_season_prior_vs_pitcher_hand", "batter_hit_1plus_rate_season_prior_vs_rhp", "batter_hit_1plus_rate_season_prior_vs_lhp"),
        ("batter_ba_season_prior_vs_pitcher_hand", "batter_ba_season_prior_vs_rhp", "batter_ba_season_prior_vs_lhp"),
        ("batter_k_rate_season_prior_vs_pitcher_hand", "batter_k_rate_season_prior_vs_rhp", "batter_k_rate_season_prior_vs_lhp"),
        ("batter_bb_rate_season_prior_vs_pitcher_hand", "batter_bb_rate_season_prior_vs_rhp", "batter_bb_rate_season_prior_vs_lhp"),
        ("batter_tb_per_ab_season_prior_vs_pitcher_hand", "batter_tb_per_ab_season_prior_vs_rhp", "batter_tb_per_ab_season_prior_vs_lhp"),
    ]

    for out_col, rhp_col, lhp_col in pairs:
        g[out_col] = np.where(is_rhp, g[rhp_col], np.where(is_lhp, g[lhp_col], np.nan))

    return g


def main() -> None:
    args = parse_args()

    df = read_any(args.input).copy()

    if "game_date" not in df.columns:
        raise ValueError("Missing game_date column.")
    if "batter_id" not in df.columns:
        raise ValueError("Missing batter_id column.")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    if "opp_starter_throw_norm" not in df.columns:
        if "opp_starter_throw" in df.columns:
            df["opp_starter_throw_norm"] = df["opp_starter_throw"].map(normalize_hand)
        else:
            df["opp_starter_throw_norm"] = "U"

    pieces = []
    groups = df.groupby("batter_id", sort=False)

    iterator = groups
    if not args.no_progress:
        iterator = tqdm(groups, total=df["batter_id"].nunique(), desc="Building platoon splits")

    for _, g in iterator:
        pieces.append(add_group_features(g))

    out = pd.concat(pieces, ignore_index=True)

    new_cols = [
        c for c in out.columns
        if (
            c.startswith("batter_games_prior_vs_")
            or c.startswith("batter_hit_1plus_rate_prior_vs_")
            or c.startswith("batter_ba_prior_vs_")
            or c.startswith("batter_k_rate_prior_vs_")
            or c.startswith("batter_bb_rate_prior_vs_")
            or c.startswith("batter_tb_per_ab_prior_vs_")
            or c.startswith("batter_games_season_prior_vs_")
            or c.startswith("batter_hit_1plus_rate_season_prior_vs_")
            or c.startswith("batter_ba_season_prior_vs_")
            or c.startswith("batter_k_rate_season_prior_vs_")
            or c.startswith("batter_bb_rate_season_prior_vs_")
            or c.startswith("batter_tb_per_ab_season_prior_vs_")
        )
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        out.to_csv(csv_path, index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": args.input,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(out)),
        "batters": int(out["batter_id"].nunique()),
        "new_column_count": int(len(new_cols)),
        "new_columns": new_cols,
        "opp_starter_throw_norm_counts": out["opp_starter_throw_norm"].value_counts(dropna=False).to_dict(),
        "non_null_rates": {c: float(out[c].notna().mean()) for c in new_cols},
        "note": "All platoon split features are prior-game only and pregame-safe.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
