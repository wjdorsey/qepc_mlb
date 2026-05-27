#!/usr/bin/env python3
"""
Add opposing starter prior-season Savant/contact-quality features to batter-game rows.

Inputs:
  --batters           batter table, usually batter_game_logs_opp_pitcher_matchup.parquet
  --pitchers_enriched pitcher_game_logs_enriched.parquet with pitcher_id + season + savant_* columns

Output:
  batter table with opp_starter_savant_*_season_prior features.

Safety:
  Uses pitcher Savant features from the PRIOR season only.
  For a 2025 batter row, this joins the opposing starter's 2024 Savant profile.
  This avoids same-season full-year leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd


SAFETY_VERSION = "batter_opp_starter_prior_savant_quality_v1_pregame_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batters", required=True)
    p.add_argument("--pitchers_enriched", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--also_csv", action="store_true")
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


def ensure_date_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    elif "date" in df.columns:
        raw = df["date"].astype("string").str.replace(r"\.0$", "", regex=True)
        df["game_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")

    if "season" not in df.columns:
        if "game_date" not in df.columns:
            raise ValueError("Missing season and game_date/date columns.")
        df["season"] = df["game_date"].dt.year

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def main() -> None:
    args = parse_args()

    bat = ensure_date_season(read_any(args.batters))
    pit = ensure_date_season(read_any(args.pitchers_enriched))

    if "opp_starter_id" not in bat.columns:
        raise ValueError("Batter table missing opp_starter_id column.")

    pitcher_id_col = first_existing(pit, ["pitcher_id", "id"])
    if pitcher_id_col is None:
        raise ValueError("Pitcher enriched table missing pitcher_id/id column.")

    savant_cols = [
        c for c in pit.columns
        if c.startswith("savant_")
        and c not in {"savant_team_id"}
    ]

    if not savant_cols:
        raise ValueError("No savant_* columns found in pitcher enriched table.")

    # Keep numeric Savant columns only.
    numeric_savant_cols = []
    for c in savant_cols:
        s = pd.to_numeric(pit[c], errors="coerce")
        if s.notna().any():
            pit[c] = s
            numeric_savant_cols.append(c)

    if not numeric_savant_cols:
        raise ValueError("No numeric Savant columns with observed values found.")

    ctx = pit[[pitcher_id_col, "season"] + numeric_savant_cols].copy()
    ctx = ctx.rename(columns={pitcher_id_col: "opp_starter_id"})
    ctx["opp_starter_id"] = ctx["opp_starter_id"].astype("string")
    ctx["season"] = pd.to_numeric(ctx["season"], errors="coerce").astype("Int64")

    # Only keep pitcher-seasons that actually have some Savant data.
    ctx["_savant_non_null_count"] = ctx[numeric_savant_cols].notna().sum(axis=1)
    ctx = ctx[ctx["_savant_non_null_count"] > 0].copy()

    # Collapse repeated pitcher-game rows to one pitcher-season row.
    agg = {c: "first" for c in numeric_savant_cols}
    ctx = (
        ctx.groupby(["opp_starter_id", "season"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    # A pitcher season S profile is used for batter rows in season S+1.
    ctx["join_season"] = ctx["season"].astype("Int64") + 1

    rename_map = {
        c: f"opp_starter_{c}_season_prior"
        for c in numeric_savant_cols
    }
    ctx = ctx.rename(columns=rename_map)

    keep_cols = ["opp_starter_id", "join_season"] + list(rename_map.values())
    ctx = ctx[keep_cols].copy()

    bat = bat.copy()
    bat["_opp_starter_id_key"] = bat["opp_starter_id"].astype("string")
    bat["_join_season"] = pd.to_numeric(bat["season"], errors="coerce").astype("Int64")

    ctx = ctx.rename(
        columns={
            "opp_starter_id": "_opp_starter_id_key",
            "join_season": "_join_season",
        }
    )

    before_rows = len(bat)
    before_cols = set(bat.columns)

    merged = bat.merge(
        ctx,
        how="left",
        on=["_opp_starter_id_key", "_join_season"],
    )

    if len(merged) != before_rows:
        raise ValueError(f"Row count changed after merge: before={before_rows}, after={len(merged)}")

    new_savant_cols = list(rename_map.values())
    merged["opp_starter_savant_context_season_prior"] = merged[new_savant_cols].notna().any(axis=1).astype(int)

    merged = merged.drop(columns=["_opp_starter_id_key", "_join_season"], errors="ignore")

    duplicate_cols = merged.columns[merged.columns.duplicated()].tolist()
    if duplicate_cols:
        print("Dropping duplicate columns before save:", duplicate_cols)
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        merged.to_csv(csv_path, index=False)

    added_cols = [c for c in merged.columns if c not in before_cols]

    summary = {
        "safety_version": SAFETY_VERSION,
        "batters_input": args.batters,
        "pitchers_enriched_input": args.pitchers_enriched,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(merged)),
        "seasons": [int(x) for x in sorted(merged["season"].dropna().astype(int).unique().tolist())],
        "pitcher_season_context_rows": int(len(ctx)),
        "savant_source_columns": numeric_savant_cols,
        "added_column_count": int(len(added_cols)),
        "added_columns": added_cols,
        "prior_savant_context_match_count": int(merged["opp_starter_savant_context_season_prior"].sum()),
        "prior_savant_context_match_rate": float(merged["opp_starter_savant_context_season_prior"].mean()),
        "match_rate_by_season": {
            str(int(season)): float(g["opp_starter_savant_context_season_prior"].mean())
            for season, g in merged.groupby("season", dropna=True)
        },
        "pregame_safety_note": "Only prior-season pitcher Savant profiles are joined. Same-season full-year Savant data is not used.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
