#!/usr/bin/env python3
"""
Build opposing starting pitcher context for the Batter 1+ Hit model.

Input:
  - batter_game_logs.parquet
  - pitcher_game_logs.parquet or raw Retrosheet pitching.csv
  - optional allplayers.csv for pitcher name/throw hand

Output:
  - batter_game_logs_opp_pitcher.parquet

Safety:
  Uses only opposing starter features calculated from PRIOR pitcher games.
  Same-game pitcher outcomes are never used as model features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SAFETY_VERSION = "batter_opp_starter_context_v1_pregame_safe"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batters", required=True)
    p.add_argument("--pitchers", required=True)
    p.add_argument("--allplayers", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--min_season", type=int, default=None)
    p.add_argument("--max_season", type=int, default=None)
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def read_any(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def coalesce_numeric(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype="float64")
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            out = out.where(out.notna(), s)
    return out


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


def normalize_pitcher_logs(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p = ensure_game_date(p)

    # Raw Retrosheet pitching.csv has duplicate value/official rows.
    if "stattype" in p.columns:
        p = p[p["stattype"].astype("string").str.lower().eq("value")].copy()

    game_col = first_existing(p, ["game_id", "gid"])
    pitcher_col = first_existing(p, ["pitcher_id", "id"])
    team_col = first_existing(p, ["team", "pitcher_team"])
    opp_col = first_existing(p, ["opponent", "opp"])

    if game_col is None or pitcher_col is None or team_col is None:
        raise ValueError("Pitcher logs missing game/team/pitcher id columns.")

    out = pd.DataFrame(index=p.index)
    out["game_id"] = p[game_col].astype("string")
    out["pitcher_id"] = p[pitcher_col].astype("string")
    out["pitcher_team"] = p[team_col].astype("string")
    out["pitcher_opponent"] = p[opp_col].astype("string") if opp_col else pd.NA
    out["game_date"] = p["game_date"]
    out["season"] = p["season"]

    if "is_starter" in p.columns:
        out["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(float).ne(0)
    elif "p_gs" in p.columns:
        out["is_starter"] = pd.to_numeric(p["p_gs"], errors="coerce").fillna(0).astype(float).ne(0)
    else:
        out["is_starter"] = False

    # Same-game pitcher stats: used only to create PRIOR rolling features.
    out["strikeouts"] = coalesce_numeric(p, ["strikeouts", "p_k"])
    out["batters_faced"] = coalesce_numeric(p, ["batters_faced", "p_bfp"])
    out["hits_allowed"] = coalesce_numeric(p, ["hits_allowed", "p_h"])
    out["walks_allowed"] = coalesce_numeric(p, ["walks", "walks_allowed", "p_w"])
    out["runs_allowed"] = coalesce_numeric(p, ["runs_allowed", "p_r"])
    out["earned_runs"] = coalesce_numeric(p, ["earned_runs", "p_er"])
    out["home_runs_allowed"] = coalesce_numeric(p, ["home_runs_allowed", "p_hr"])
    out["outs_recorded"] = coalesce_numeric(p, ["outs_recorded", "p_ipouts"])

    if "innings_pitched" in p.columns:
        out["innings_pitched"] = pd.to_numeric(p["innings_pitched"], errors="coerce")
    else:
        out["innings_pitched"] = out["outs_recorded"] / 3.0

    out = out.dropna(subset=["game_id", "pitcher_id", "pitcher_team", "game_date"]).copy()
    out = out[out["is_starter"]].copy()

    return out


def attach_pitcher_names(p: pd.DataFrame, allplayers_path: Optional[str]) -> pd.DataFrame:
    if not allplayers_path:
        p["opp_starter_name"] = pd.NA
        p["opp_starter_throw"] = pd.NA
        return p

    path = Path(allplayers_path)
    if not path.exists():
        p["opp_starter_name"] = pd.NA
        p["opp_starter_throw"] = pd.NA
        return p

    ap = pd.read_csv(path, usecols=lambda c: c in {"id", "last", "first", "throw", "team", "season"})
    ap["pitcher_id"] = ap["id"].astype("string")
    ap["pitcher_team"] = ap["team"].astype("string")
    ap["season"] = pd.to_numeric(ap["season"], errors="coerce").astype("Int64")
    ap["opp_starter_name"] = (
        ap["first"].astype("string").fillna("").str.strip()
        + " "
        + ap["last"].astype("string").fillna("").str.strip()
    ).str.strip()
    ap["opp_starter_throw"] = ap["throw"].astype("string")

    ap = ap[["pitcher_id", "pitcher_team", "season", "opp_starter_name", "opp_starter_throw"]].drop_duplicates(
        ["pitcher_id", "pitcher_team", "season"]
    )

    return p.merge(ap, how="left", on=["pitcher_id", "pitcher_team", "season"])


def add_prior_pitcher_features(p: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    p = p.sort_values(["pitcher_id", "game_date", "game_id"]).copy()

    stats = [
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

    rate_specs = {
        "k_per_bf": ("strikeouts", "batters_faced"),
        "hits_per_bf": ("hits_allowed", "batters_faced"),
        "bb_per_bf": ("walks_allowed", "batters_faced"),
        "hr_per_bf": ("home_runs_allowed", "batters_faced"),
        "er_per_ip": ("earned_runs", "innings_pitched"),
    }

    pieces = []
    iterator = p.groupby("pitcher_id", sort=False)
    if show_progress:
        iterator = tqdm(iterator, total=p["pitcher_id"].nunique(), desc="Building opposing starter prior features")

    for _, g in iterator:
        g = g.sort_values(["game_date", "game_id"]).copy()
        g["opp_starter_games_prior"] = np.arange(len(g), dtype=float)
        g["opp_starter_days_rest"] = g["game_date"].diff().dt.days

        for s in stats:
            shifted = pd.to_numeric(g[s], errors="coerce").shift(1)
            g[f"opp_starter_{s}_lag1"] = shifted
            for w in [3, 5, 10]:
                g[f"opp_starter_{s}_roll{w}"] = shifted.rolling(w, min_periods=1).mean()

        for rate_name, (num, den) in rate_specs.items():
            num_shift = pd.to_numeric(g[num], errors="coerce").shift(1)
            den_shift = pd.to_numeric(g[den], errors="coerce").shift(1).replace(0, np.nan)

            g[f"opp_starter_{rate_name}_lag1"] = num_shift / den_shift
            for w in [3, 5, 10]:
                g[f"opp_starter_{rate_name}_roll{w}"] = (
                    num_shift.rolling(w, min_periods=1).sum()
                    / den_shift.rolling(w, min_periods=1).sum().replace(0, np.nan)
                )

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def main():
    args = parse_args()

    bat = read_any(args.batters)
    pit_raw = read_any(args.pitchers)

    bat = ensure_game_date(bat)

    if args.min_season is not None:
        bat = bat[bat["season"].astype(float) >= args.min_season].copy()
    if args.max_season is not None:
        bat = bat[bat["season"].astype(float) <= args.max_season].copy()

    pit = normalize_pitcher_logs(pit_raw)
    pit = attach_pitcher_names(pit, args.allplayers)
    pit = add_prior_pitcher_features(pit, show_progress=not args.no_progress)

    # One opposing starter context row per game/team.
    context_cols = [
        "game_id",
        "pitcher_team",
        "pitcher_id",
        "opp_starter_name",
        "opp_starter_throw",
    ] + [c for c in pit.columns if c.startswith("opp_starter_")]

    ctx = pit[context_cols].copy()
    ctx = ctx.rename(
        columns={
            "pitcher_team": "_opp_team_key",
            "pitcher_id": "opp_starter_id",
        }
    )
    ctx["_game_key"] = ctx["game_id"].astype("string")
    ctx["_opp_team_key"] = ctx["_opp_team_key"].astype("string")

    # Critical safety/shape guard:
    # There must be exactly one opposing-starter context row per game + team.
    # If duplicates exist, a merge would duplicate batter rows and inflate the dataset.
    context_key_cols = ["_game_key", "_opp_team_key"]
    duplicate_context_rows = int(ctx.duplicated(context_key_cols, keep=False).sum())
    duplicate_context_keys = int(ctx[context_key_cols].duplicated().sum())

    if duplicate_context_rows:
        print(
            f"Found duplicate opposing-starter context rows before merge: "
            f"{duplicate_context_rows} rows across {duplicate_context_keys} duplicate keys. "
            f"Dropping duplicate keys with keep='first'."
        )

    ctx = ctx.sort_values(context_key_cols + ["opp_starter_games_prior"], na_position="last")
    ctx = ctx.drop_duplicates(context_key_cols, keep="first").copy()

    ctx = ctx.drop(columns=["game_id"])

    game_col = first_existing(bat, ["game_id", "gid"])
    opp_col = first_existing(bat, ["opponent", "opp"])

    if game_col is None or opp_col is None:
        raise ValueError("Batter logs missing game_id/gid or opponent/opp columns.")

    bat["_game_key"] = bat[game_col].astype("string")
    bat["_opp_team_key"] = bat[opp_col].astype("string")

    before_cols = set(bat.columns)
    merged = bat.merge(ctx, how="left", on=["_game_key", "_opp_team_key"])

    merged["has_opp_starter_context"] = merged["opp_starter_id"].notna()

    # Parquet/pyarrow cannot save duplicate column names.
    # This can happen because opp_starter_name / opp_starter_throw are explicit
    # context columns and also match the opp_starter_* prefix.
    duplicate_cols = merged.columns[merged.columns.duplicated()].tolist()
    if duplicate_cols:
        print("Dropping duplicate columns before save:", duplicate_cols)
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    merged = merged.drop(columns=["_game_key", "_opp_team_key"], errors="ignore")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        merged.to_csv(csv_path, index=False)

    added_cols = [c for c in merged.columns if c not in before_cols]
    feature_cols = [c for c in added_cols if c.startswith("opp_starter_")] + ["has_opp_starter_context"]

    summary = {
        "safety_version": SAFETY_VERSION,
        "batters_input": args.batters,
        "pitchers_input": args.pitchers,
        "allplayers_input": args.allplayers,
        "rows": int(len(merged)),
        "date_min": str(merged["game_date"].min().date()) if merged["game_date"].notna().any() else None,
        "date_max": str(merged["game_date"].max().date()) if merged["game_date"].notna().any() else None,
        "seasons": [int(x) for x in sorted(merged["season"].dropna().astype(int).unique().tolist())],
        "opp_starter_match_count": int(merged["has_opp_starter_context"].sum()),
        "opp_starter_match_rate": float(merged["has_opp_starter_context"].mean()),
        "added_column_count": int(len(added_cols)),
        "opp_starter_feature_columns": feature_cols,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "pregame_safety_note": "Only prior pitcher games are used for opp_starter_* rolling/lag features.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("saved parquet:", out_path)
    if csv_path:
        print("saved csv:", csv_path)
    print("saved summary:", summary_path)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
