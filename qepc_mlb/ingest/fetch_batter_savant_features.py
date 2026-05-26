#!/usr/bin/env python3
"""
Fetch batter-side Baseball Savant / Statcast feature tables through pybaseball.

Sources:
  - statcast_batter_expected_stats
  - statcast_batter_exitvelo_barrels
  - statcast_batter_percentile_ranks

Output:
  cache/mlb/processed/batter_savant_features.parquet

Notes:
  - This is season-level batter process/contact-quality data.
  - We will join it safely later, likely as prior-season features first.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SAFETY_VERSION = "fetch_batter_savant_features_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start_year", type=int, default=2022)
    p.add_argument("--end_year", type=int, default=2025)
    p.add_argument("--raw_dir", default="cache/mlb/raw/savant/batter")
    p.add_argument("--out", default="cache/mlb/processed/batter_savant_features.parquet")
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--min_pa", type=int, default=1)
    p.add_argument("--min_bbe", type=int, default=1)
    p.add_argument("--sleep", type=float, default=1.0)
    return p.parse_args()


def normalize_col(c: Any) -> str:
    s = str(c).strip().lower()
    s = s.replace("%", "pct")
    s = s.replace("+", "plus")
    s = s.replace("-", "_")
    s = s.replace("/", "_per_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_player_name(df: pd.DataFrame) -> pd.Series:
    if "player_name" in df.columns:
        return df["player_name"].astype("string")
    if "name" in df.columns:
        return df["name"].astype("string")
    if "last_name" in df.columns and "first_name" in df.columns:
        return (df["first_name"].astype("string").fillna("") + " " + df["last_name"].astype("string").fillna("")).str.strip()
    if "last_name_first_name" in df.columns:
        return df["last_name_first_name"].astype("string")
    return pd.Series(pd.NA, index=df.index, dtype="string")


def prep_table(df: pd.DataFrame, *, year: int, prefix: str) -> pd.DataFrame:
    df = normalize_columns(df)

    id_col = first_existing(
        df,
        [
            "player_id",
            "playerid",
            "player_id_",
            "mlbam_id",
            "mlb_id",
            "id",
        ],
    )

    if id_col is None:
        raise ValueError(f"Could not find player id column for {prefix} {year}. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    out["season"] = int(year)
    out["batter_name"] = make_player_name(df)

    # Keep numeric feature columns only. Raw CSVs are saved separately.
    skip = {
        id_col,
        "player_id",
        "playerid",
        "player_id_",
        "mlbam_id",
        "mlb_id",
        "id",
        "player_name",
        "name",
        "last_name",
        "first_name",
        "last_name_first_name",
        "season",
        "year",
    }

    for c in df.columns:
        if c in skip:
            continue

        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            out[f"batter_savant_{prefix}_{c}"] = numeric

    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    # Collapse any duplicate player rows safely.
    feature_cols = [c for c in out.columns if c not in {"player_id", "season", "batter_name"}]
    agg = {c: "first" for c in feature_cols}
    agg["batter_name"] = "first"

    out = (
        out.groupby(["player_id", "season"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    return out


def fetch_year(year: int, raw_dir: Path, min_pa: int, min_bbe: int, sleep: float) -> Dict[str, pd.DataFrame]:
    try:
        from pybaseball import (
            statcast_batter_expected_stats,
            statcast_batter_exitvelo_barrels,
            statcast_batter_percentile_ranks,
        )
    except Exception as e:
        raise RuntimeError(
            "Could not import pybaseball. Run: python -m pip install pybaseball"
        ) from e

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching batter Savant tables for {year}...")

    tables: Dict[str, pd.DataFrame] = {}

    print(f"  expected stats, min_pa={min_pa}")
    expected = statcast_batter_expected_stats(year, minPA=min_pa)
    expected.to_csv(raw_dir / f"savant_batter_expected_{year}.csv", index=False)
    tables["expected"] = prep_table(expected, year=year, prefix="expected")
    time.sleep(sleep)

    print(f"  exit velo/barrels, min_bbe={min_bbe}")
    ev = statcast_batter_exitvelo_barrels(year, minBBE=min_bbe)
    ev.to_csv(raw_dir / f"savant_batter_exitvelo_barrels_{year}.csv", index=False)
    tables["exitvelo"] = prep_table(ev, year=year, prefix="ev")
    time.sleep(sleep)

    print("  percentile ranks")
    pct = statcast_batter_percentile_ranks(year)
    pct.to_csv(raw_dir / f"savant_batter_percentile_{year}.csv", index=False)
    tables["percentile"] = prep_table(pct, year=year, prefix="pct")
    time.sleep(sleep)

    return tables


def merge_year_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None

    for name, df in tables.items():
        if merged is None:
            merged = df
            continue

        # Avoid duplicate batter_name columns during merge.
        df2 = df.drop(columns=["batter_name"], errors="ignore")
        merged = merged.merge(df2, how="outer", on=["player_id", "season"])

    if merged is None:
        return pd.DataFrame()

    return merged


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_years: List[pd.DataFrame] = []
    year_summaries: Dict[str, Any] = {}

    for year in range(args.start_year, args.end_year + 1):
        tables = fetch_year(year, raw_dir, args.min_pa, args.min_bbe, args.sleep)
        merged_year = merge_year_tables(tables)
        all_years.append(merged_year)

        year_summaries[str(year)] = {
            "expected_rows": int(len(tables["expected"])),
            "exitvelo_rows": int(len(tables["exitvelo"])),
            "percentile_rows": int(len(tables["percentile"])),
            "merged_rows": int(len(merged_year)),
            "feature_columns": int(len([c for c in merged_year.columns if c.startswith("batter_savant_")])),
        }

    out = pd.concat(all_years, ignore_index=True) if all_years else pd.DataFrame()

    # Remove all-empty feature columns.
    feature_cols = [c for c in out.columns if c.startswith("batter_savant_")]
    nonempty_features = [c for c in feature_cols if pd.to_numeric(out[c], errors="coerce").notna().any()]
    keep_cols = ["player_id", "season", "batter_name"] + nonempty_features
    out = out[keep_cols].copy()

    out.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        out.to_csv(csv_path, index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "raw_dir": str(raw_dir),
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(out)),
        "seasons": [int(x) for x in sorted(out["season"].dropna().unique().tolist())] if len(out) else [],
        "players": int(out["player_id"].nunique()) if len(out) else 0,
        "feature_column_count": int(len(nonempty_features)),
        "year_summaries": year_summaries,
        "sample_feature_columns": nonempty_features[:50],
        "note": "Season-level batter Savant features fetched. Join safely later as prior-season context unless building a diagnostic-only same-season experiment.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
