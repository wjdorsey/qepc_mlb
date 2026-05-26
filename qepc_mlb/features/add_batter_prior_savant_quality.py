#!/usr/bin/env python3
"""
Add batter prior-season Savant/contact-quality features to batter-game rows.

Safety:
  Uses batter Savant profile from the PRIOR season only.
  For a 2025 batter row, joins the batter's 2024 Savant profile.
  This avoids same-season full-year leakage.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


SAFETY_VERSION = "batter_prior_savant_quality_v1_pregame_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batters", required=True)
    p.add_argument("--batter_savant", required=True)
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


def normalize_name(x: Any) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # Baseball Savant / pybaseball often uses "Last, First".
    # Retrosheet-style batter logs usually use "First Last".
    # Convert before stripping punctuation:
    #   "Ramírez, José" -> "José Ramírez"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace(".", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "season" not in df.columns:
        if "game_date" not in df.columns:
            raise ValueError("Missing season and game_date columns.")
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["season"] = df["game_date"].dt.year

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def main() -> None:
    args = parse_args()

    bat = ensure_season(read_any(args.batters))
    sav = ensure_season(read_any(args.batter_savant))

    if "batter_name" not in bat.columns:
        raise ValueError("Batter table missing batter_name column.")
    if "batter_name" not in sav.columns:
        raise ValueError("Batter Savant table missing batter_name column.")

    savant_cols = [c for c in sav.columns if c.startswith("batter_savant_")]
    if not savant_cols:
        raise ValueError("No batter_savant_* columns found.")

    # Keep only numeric Savant columns with at least one observed value.
    numeric_cols = []
    for c in savant_cols:
        s = pd.to_numeric(sav[c], errors="coerce")
        if s.notna().any():
            sav[c] = s
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric batter_savant_* columns found.")

    sav = sav.copy()
    sav["_batter_name_key"] = sav["batter_name"].map(normalize_name)
    sav = sav[sav["_batter_name_key"].ne("")].copy()

    # Collapse to one player-name/season row.
    agg = {c: "first" for c in numeric_cols}
    if "player_id" in sav.columns:
        agg["player_id"] = "first"
    agg["batter_name"] = "first"

    ctx = (
        sav.groupby(["_batter_name_key", "season"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    # Savant season S is used for batter rows in season S+1.
    ctx["_join_season"] = ctx["season"].astype("Int64") + 1

    rename_map = {c: f"{c}_season_prior" for c in numeric_cols}
    ctx = ctx.rename(columns=rename_map)

    keep_cols = ["_batter_name_key", "_join_season"] + list(rename_map.values())
    if "player_id" in ctx.columns:
        ctx = ctx.rename(columns={"player_id": "batter_savant_player_id_season_prior"})
        keep_cols.append("batter_savant_player_id_season_prior")

    ctx = ctx[keep_cols].copy()
    ctx = ctx.drop_duplicates(["_batter_name_key", "_join_season"], keep="first")

    bat = bat.copy()
    bat["_batter_name_key"] = bat["batter_name"].map(normalize_name)
    bat["_join_season"] = pd.to_numeric(bat["season"], errors="coerce").astype("Int64")

    before_rows = len(bat)
    before_cols = set(bat.columns)

    merged = bat.merge(
        ctx,
        how="left",
        on=["_batter_name_key", "_join_season"],
    )

    if len(merged) != before_rows:
        raise ValueError(f"Row count changed after merge: before={before_rows}, after={len(merged)}")

    new_feature_cols = list(rename_map.values())
    merged["batter_savant_context_season_prior"] = merged[new_feature_cols].notna().any(axis=1).astype(int)

    merged = merged.drop(columns=["_batter_name_key", "_join_season"], errors="ignore")

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
        "batter_savant_input": args.batter_savant,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(merged)),
        "seasons": [int(x) for x in sorted(merged["season"].dropna().astype(int).unique().tolist())],
        "savant_context_rows": int(len(ctx)),
        "savant_source_columns": numeric_cols,
        "added_column_count": int(len(added_cols)),
        "prior_batter_savant_match_count": int(merged["batter_savant_context_season_prior"].sum()),
        "prior_batter_savant_match_rate": float(merged["batter_savant_context_season_prior"].mean()),
        "match_rate_by_season": {
            str(int(season)): float(g["batter_savant_context_season_prior"].mean())
            for season, g in merged.groupby("season", dropna=True)
        },
        "pregame_safety_note": "Only prior-season batter Savant profiles are joined. Same-season full-year Savant data is not used.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
