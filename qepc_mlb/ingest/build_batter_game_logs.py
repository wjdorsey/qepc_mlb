#!/usr/bin/env python3
"""
Build QEPC-MLB batter game logs for a hitter 1+ hit model.

Inputs are Retrosheet CSV exports, primarily batting.csv, with optional
allplayers.csv, gameinfo.csv, and teamstats.csv enrichment.

Output is one row per batter-game/team appearance with a binary target:
    hit_1plus = 1 if hits >= 1 else 0

Important: this table intentionally keeps same-game outcomes such as hits,
at_bats, and plate_appearances because they define the target and diagnostics.
Pregame model scripts should use only lagged/rolling/context fields, not the
same-game outcome columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SAFETY_VERSION = "batter_game_logs_v1_retrosheet_hit_1plus"

DEFAULT_BATTING = Path("cache/mlb/raw/retrosheet/batting.csv")
DEFAULT_ALLPLAYERS = Path("cache/mlb/raw/retrosheet/allplayers.csv")
DEFAULT_GAMEINFO = Path("cache/mlb/raw/retrosheet/gameinfo.csv")
DEFAULT_TEAMSTATS = Path("cache/mlb/raw/retrosheet/teamstats.csv")
DEFAULT_OUT = Path("cache/mlb/processed/batter_game_logs.parquet")

BATTING_USECOLS = [
    "gid",
    "id",
    "team",
    "b_lp",
    "b_seq",
    "stattype",
    "b_pa",
    "b_ab",
    "b_r",
    "b_h",
    "b_d",
    "b_t",
    "b_hr",
    "b_rbi",
    "b_sh",
    "b_sf",
    "b_hbp",
    "b_w",
    "b_iw",
    "b_k",
    "b_sb",
    "b_cs",
    "b_gdp",
    "b_xi",
    "b_roe",
    "dh",
    "ph",
    "pr",
    "date",
    "number",
    "site",
    "vishome",
    "opp",
    "win",
    "loss",
    "tie",
    "gametype",
]

NUMERIC_RENAMES = {
    "b_pa": "plate_appearances",
    "b_ab": "at_bats",
    "b_r": "runs",
    "b_h": "hits",
    "b_d": "doubles",
    "b_t": "triples",
    "b_hr": "home_runs",
    "b_rbi": "rbi",
    "b_sh": "sac_hits",
    "b_sf": "sac_flies",
    "b_hbp": "hit_by_pitch",
    "b_w": "walks",
    "b_iw": "intentional_walks",
    "b_k": "strikeouts",
    "b_sb": "stolen_bases",
    "b_cs": "caught_stealing",
    "b_gdp": "grounded_double_plays",
    "b_xi": "interference_reached",
    "b_roe": "reached_on_error",
    "dh": "dh_flag",
    "ph": "pinch_hit_flag",
    "pr": "pinch_run_flag",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_retro_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def normalize_boolish_numeric(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    return num.fillna(0).gt(0)


def read_csv_usecols(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path, low_memory=False)
    # Some Retrosheet exports can vary. Only request columns that exist.
    header = pd.read_csv(path, nrows=0)
    available = [c for c in usecols if c in header.columns]
    return pd.read_csv(path, usecols=available, low_memory=False)


def load_batting(path: Path, min_season: Optional[int], max_season: Optional[int], include_non_regular: bool) -> pd.DataFrame:
    print(f"Loading batting source: {path}")
    df = read_csv_usecols(path, BATTING_USECOLS)

    if "stattype" in df.columns:
        before = len(df)
        df = df[df["stattype"].astype("string").str.lower().eq("value")].copy()
        print(f"Kept stattype=value rows: {len(df):,} / {before:,}")

    if not include_non_regular and "gametype" in df.columns:
        before = len(df)
        df = df[df["gametype"].astype("string").str.lower().eq("regular")].copy()
        print(f"Kept regular-season rows: {len(df):,} / {before:,}")

    df = df.rename(
        columns={
            "gid": "game_id",
            "id": "batter_id",
            "b_lp": "lineup_position_raw",
            "b_seq": "batting_sequence",
            "opp": "opponent",
            "vishome": "home_away",
            **NUMERIC_RENAMES,
        }
    )

    df["game_date"] = parse_retro_date(df["date"])
    df["season"] = df["game_date"].dt.year.astype("Int64")

    if min_season is not None:
        df = df[df["season"].ge(min_season)].copy()
    if max_season is not None:
        df = df[df["season"].le(max_season)].copy()

    for col in NUMERIC_RENAMES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["lineup_position_raw", "batting_sequence", "win", "loss", "tie"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_away"] = df["home_away"].astype("string").str.lower().map({"h": "home", "v": "away"}).fillna(df["home_away"].astype("string"))
    df["is_home"] = df["home_away"].eq("home")

    # Core target/outcome features.
    df["singles"] = (df.get("hits", 0) - df.get("doubles", 0) - df.get("triples", 0) - df.get("home_runs", 0)).clip(lower=0)
    df["total_bases"] = df["singles"] + 2 * df.get("doubles", 0) + 3 * df.get("triples", 0) + 4 * df.get("home_runs", 0)
    df["hit_1plus"] = df.get("hits", 0).gt(0).astype(int)
    df["on_base_events"] = df.get("hits", 0) + df.get("walks", 0) + df.get("hit_by_pitch", 0)

    # Retrosheet lineup position can be 1-9 for starters and other values/noise for substitutes.
    df["lineup_slot"] = pd.to_numeric(df.get("lineup_position_raw"), errors="coerce")
    df.loc[~df["lineup_slot"].between(1, 9), "lineup_slot"] = np.nan
    df["is_starting_lineup_by_batting_file"] = df["batting_sequence"].fillna(1).eq(1) & df["lineup_slot"].between(1, 9)

    # Stable sort for rolling features.
    sort_cols = ["batter_id", "game_date", "game_id", "team", "batting_sequence"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def join_allplayers(df: pd.DataFrame, path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        print("allplayers not supplied/found; batter_name and batter_hand will be missing.")
        df["batter_name"] = pd.NA
        df["batter_hand"] = pd.NA
        return df

    print(f"Joining allplayers: {path}")
    usecols = ["id", "last", "first", "bat", "throw", "team", "season"]
    ap = read_csv_usecols(path, usecols)
    ap = ap.rename(columns={"id": "batter_id", "bat": "batter_hand", "throw": "throw_hand"})
    ap["season"] = pd.to_numeric(ap["season"], errors="coerce").astype("Int64")
    ap["batter_name"] = (ap["first"].astype("string").fillna("") + " " + ap["last"].astype("string").fillna("")).str.strip()
    ap["batter_hand"] = ap["batter_hand"].astype("string").str.upper().replace({"": pd.NA})

    exact_cols = ["batter_id", "season", "team"]
    ap_exact = ap[exact_cols + ["batter_name", "batter_hand", "throw_hand"]].drop_duplicates(exact_cols)
    out = df.merge(ap_exact, on=exact_cols, how="left")

    # Fallback for traded players or team-code quirks.
    missing = out["batter_name"].isna() | out["batter_hand"].isna()
    if missing.any():
        ap_fallback = (
            ap[["batter_id", "season", "batter_name", "batter_hand", "throw_hand"]]
            .dropna(subset=["batter_id", "season"])
            .drop_duplicates(["batter_id", "season"])
        )
        fb = out.loc[missing, ["batter_id", "season"]].merge(ap_fallback, on=["batter_id", "season"], how="left", suffixes=("", "_fb"))
        out.loc[missing, "batter_name"] = out.loc[missing, "batter_name"].fillna(fb["batter_name"].to_numpy())
        out.loc[missing, "batter_hand"] = out.loc[missing, "batter_hand"].fillna(fb["batter_hand"].to_numpy())
        if "throw_hand" in out.columns and "throw_hand" in fb.columns:
            out.loc[missing, "throw_hand"] = out.loc[missing, "throw_hand"].fillna(fb["throw_hand"].to_numpy())

    return out


def join_gameinfo(df: pd.DataFrame, path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        print("gameinfo not supplied/found; environment fields will be missing.")
        return df

    print(f"Joining gameinfo: {path}")
    usecols = [
        "gid", "visteam", "hometeam", "site", "daynight", "usedh", "temp", "winddir", "windspeed", "season"
    ]
    gi = read_csv_usecols(path, usecols)
    gi = gi.rename(
        columns={
            "gid": "game_id",
            "visteam": "away_team",
            "hometeam": "home_team",
            "site": "park_id",
            "daynight": "env_daynight",
            "usedh": "env_usedh",
            "temp": "env_temp",
            "winddir": "env_winddir",
            "windspeed": "env_windspeed",
        }
    )
    for col in ["env_temp", "env_windspeed"]:
        if col in gi.columns:
            gi[col] = pd.to_numeric(gi[col], errors="coerce")
            gi.loc[gi[col].lt(0), col] = np.nan
    out = df.merge(gi.drop_duplicates("game_id"), on="game_id", how="left", suffixes=("", "_gi"))
    out["has_env_context"] = out.get("park_id").notna() if "park_id" in out.columns else False
    return out


def join_teamstats_starters(df: pd.DataFrame, path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        print("teamstats not supplied/found; starting lineup flags from teamstats will be missing.")
        df["is_starting_lineup"] = df.get("is_starting_lineup_by_batting_file", False)
        return df

    print(f"Joining teamstats starter slots: {path}")
    start_cols = [f"start_l{i}" for i in range(1, 10)]
    usecols = ["gid", "team", "stattype", *start_cols]
    ts = read_csv_usecols(path, usecols)
    if "stattype" in ts.columns:
        ts = ts[ts["stattype"].astype("string").str.lower().eq("value")].copy()
    ts = ts.rename(columns={"gid": "game_id"})

    frames = []
    available = [c for c in start_cols if c in ts.columns]
    for i, col in enumerate(available, start=1):
        tmp = ts[["game_id", "team", col]].rename(columns={col: "batter_id"})
        tmp["lineup_slot_teamstats"] = i
        frames.append(tmp)
    if not frames:
        df["is_starting_lineup"] = df.get("is_starting_lineup_by_batting_file", False)
        return df

    starters = pd.concat(frames, ignore_index=True)
    starters = starters.dropna(subset=["batter_id"]).drop_duplicates(["game_id", "team", "batter_id"])
    out = df.merge(starters, on=["game_id", "team", "batter_id"], how="left")
    out["is_starting_lineup_teamstats"] = out["lineup_slot_teamstats"].notna()
    out["is_starting_lineup"] = out["is_starting_lineup_teamstats"] | out.get("is_starting_lineup_by_batting_file", False)
    out["lineup_slot"] = out["lineup_slot"].fillna(out["lineup_slot_teamstats"])
    return out


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    return np.where(den.gt(0), num / den, np.nan)


def add_rolling_features(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    print("Building leakage-safe batter rolling features...")
    df = df.sort_values(["batter_id", "game_date", "game_id", "team"]).copy()
    groups = df.groupby("batter_id", sort=False, group_keys=False)
    df["batter_games_prior"] = groups.cumcount()

    # Days rest.
    df["batter_days_rest"] = groups["game_date"].diff().dt.days

    base_cols = [
        "hit_1plus", "hits", "at_bats", "plate_appearances", "total_bases", "walks", "strikeouts", "home_runs"
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    iterator: Iterable[str] = base_cols
    if tqdm is not None and show_progress:
        iterator = tqdm(base_cols, desc="Rolling batter features")

    for col in iterator:
        shifted = groups[col].shift(1)
        df[f"{col}_lag1"] = shifted
        for w in [3, 5, 10, 20]:
            df[f"{col}_roll{w}"] = shifted.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())

    # Rate features from shifted cumulative/rolling sums.
    shifted_hits = groups["hits"].shift(1) if "hits" in df.columns else pd.Series(np.nan, index=df.index)
    shifted_ab = groups["at_bats"].shift(1) if "at_bats" in df.columns else pd.Series(np.nan, index=df.index)
    shifted_pa = groups["plate_appearances"].shift(1) if "plate_appearances" in df.columns else pd.Series(np.nan, index=df.index)
    shifted_k = groups["strikeouts"].shift(1) if "strikeouts" in df.columns else pd.Series(np.nan, index=df.index)
    shifted_bb = groups["walks"].shift(1) if "walks" in df.columns else pd.Series(np.nan, index=df.index)

    for w in [5, 10, 20]:
        hits_sum = shifted_hits.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        ab_sum = shifted_ab.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        pa_sum = shifted_pa.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        k_sum = shifted_k.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        bb_sum = shifted_bb.groupby(df["batter_id"], sort=False).transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        df[f"batter_ba_roll{w}"] = safe_div(hits_sum, ab_sum)
        df[f"batter_k_rate_roll{w}"] = safe_div(k_sum, pa_sum)
        df[f"batter_bb_rate_roll{w}"] = safe_div(bb_sum, pa_sum)

    # Season-to-date prior rates.
    for col in ["hits", "at_bats", "plate_appearances", "strikeouts", "walks"]:
        if col in df.columns:
            prior_cum = groups[col].cumsum() - df[col]
            df[f"{col}_season_prior"] = prior_cum
    if {"hits_season_prior", "at_bats_season_prior"}.issubset(df.columns):
        df["batter_ba_season_prior"] = safe_div(df["hits_season_prior"], df["at_bats_season_prior"])
    if {"strikeouts_season_prior", "plate_appearances_season_prior"}.issubset(df.columns):
        df["batter_k_rate_season_prior"] = safe_div(df["strikeouts_season_prior"], df["plate_appearances_season_prior"])
    if {"walks_season_prior", "plate_appearances_season_prior"}.issubset(df.columns):
        df["batter_bb_rate_season_prior"] = safe_div(df["walks_season_prior"], df["plate_appearances_season_prior"])

    return df


def summarize(df: pd.DataFrame, out: Path, args: argparse.Namespace) -> dict:
    return {
        "safety_version": SAFETY_VERSION,
        "rows": int(len(df)),
        "batters": int(df["batter_id"].nunique()) if "batter_id" in df else 0,
        "seasons": sorted([int(x) for x in df["season"].dropna().unique()]) if "season" in df else [],
        "date_min": str(df["game_date"].min().date()) if len(df) and "game_date" in df else None,
        "date_max": str(df["game_date"].max().date()) if len(df) and "game_date" in df else None,
        "hit_1plus_rate": float(df["hit_1plus"].mean()) if "hit_1plus" in df and len(df) else None,
        "named_rows": int(df["batter_name"].notna().sum()) if "batter_name" in df else 0,
        "handed_rows": int(df["batter_hand"].notna().sum()) if "batter_hand" in df else 0,
        "starter_rows": int(df["is_starting_lineup"].fillna(False).sum()) if "is_starting_lineup" in df else 0,
        "env_rows": int(df["has_env_context"].fillna(False).sum()) if "has_env_context" in df else 0,
        "rolling_feature_columns": [c for c in df.columns if "roll" in c or c.endswith("_lag1") or c.endswith("_season_prior")],
        "out": str(out),
        "args": vars(args),
        "pregame_warning": "Same-game outcome columns are present for targets/diagnostics. Pregame models should use lagged/rolling/context features only.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build QEPC-MLB batter game logs for 1+ hit modeling.")
    p.add_argument("--batting", default=str(DEFAULT_BATTING), help="Path to Retrosheet batting.csv")
    p.add_argument("--allplayers", default=str(DEFAULT_ALLPLAYERS), help="Path to Retrosheet allplayers.csv")
    p.add_argument("--gameinfo", default=str(DEFAULT_GAMEINFO), help="Path to Retrosheet gameinfo.csv")
    p.add_argument("--teamstats", default=str(DEFAULT_TEAMSTATS), help="Path to Retrosheet teamstats.csv")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output parquet path")
    p.add_argument("--also_csv", action="store_true", help="Also save CSV next to parquet")
    p.add_argument("--min_season", type=int, default=2018, help="Minimum season to keep")
    p.add_argument("--max_season", type=int, default=None, help="Maximum season to keep")
    p.add_argument("--include_non_regular", action="store_true", help="Keep non-regular-season games")
    p.add_argument("--no_features", action="store_true", help="Do not build lag/rolling batter features")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batting = Path(args.batting)
    allplayers = Path(args.allplayers) if args.allplayers else None
    gameinfo = Path(args.gameinfo) if args.gameinfo else None
    teamstats = Path(args.teamstats) if args.teamstats else None
    out = Path(args.out)

    if not batting.exists():
        raise FileNotFoundError(f"batting file not found: {batting}")

    df = load_batting(batting, args.min_season, args.max_season, args.include_non_regular)
    df = join_allplayers(df, allplayers)
    df = join_gameinfo(df, gameinfo)
    df = join_teamstats_starters(df, teamstats)

    if not args.no_features:
        df = add_rolling_features(df, show_progress=not args.no_progress)

    # Helpful final ordering.
    preferred = [
        "game_id", "game_date", "season", "batter_id", "batter_name", "batter_hand", "team", "opponent",
        "home_away", "is_home", "lineup_slot", "is_starting_lineup", "plate_appearances", "at_bats", "hits",
        "hit_1plus", "total_bases", "singles", "doubles", "triples", "home_runs", "walks", "strikeouts",
        "runs", "rbi", "park_id", "env_temp", "env_windspeed", "env_winddir", "env_daynight", "env_usedh",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    ensure_parent(out)
    df.to_parquet(out, index=False)
    print(f"saved parquet: {out}")

    if args.also_csv:
        csv_out = out.with_suffix(".csv")
        df.to_csv(csv_out, index=False)
        print(f"saved csv: {csv_out}")

    summary = summarize(df, out, args)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
