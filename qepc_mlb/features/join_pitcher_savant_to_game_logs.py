from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


NAME_KEY_COL = "pitcher_name_key"


def maybe_tqdm(it: Iterable, *, total: int | None = None, desc: str = "", disable: bool = False):
    if disable or tqdm is None:
        return it
    return tqdm(it, total=total, desc=desc)


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_name(value) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().lower()
    if not s:
        return pd.NA
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}".strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else pd.NA


def add_name_key(df: pd.DataFrame, source_name_col: str) -> pd.DataFrame:
    out = df.copy()
    out[NAME_KEY_COL] = out[source_name_col].map(normalize_name).astype("string")
    return out


def dedupe_savant(df: pd.DataFrame) -> pd.DataFrame:
    # one row per pitcher-season. Prefer rows with the most non-null feature values.
    core = [c for c in df.columns if c not in {"pitcher_name", NAME_KEY_COL}]
    out = df.copy()
    out["_nonnull_score"] = out[core].notna().sum(axis=1)
    out = out.sort_values(["season", NAME_KEY_COL, "_nonnull_score"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["season", NAME_KEY_COL], keep="first")
    return out.drop(columns=["_nonnull_score"])


def build_summary(joined: pd.DataFrame, savant_feature_cols: list[str]) -> dict:
    feature_nonnull = joined[savant_feature_cols].notna().any(axis=1) if savant_feature_cols else pd.Series(False, index=joined.index)
    pitcher_season = joined[["season", "pitcher_name", NAME_KEY_COL]].drop_duplicates().copy()
    ps_feature_nonnull = pitcher_season.merge(
        joined[["season", NAME_KEY_COL] + savant_feature_cols].drop_duplicates(),
        on=["season", NAME_KEY_COL],
        how="left",
    )
    ps_covered = ps_feature_nonnull[savant_feature_cols].notna().any(axis=1) if savant_feature_cols else pd.Series(False, index=ps_feature_nonnull.index)

    summary = {
        "rows": int(len(joined)),
        "seasons": sorted([int(x) for x in joined["season"].dropna().astype(int).unique().tolist()]),
        "pitchers": int(joined["pitcher_id"].nunique()) if "pitcher_id" in joined.columns else None,
        "row_match_count": int(feature_nonnull.sum()),
        "row_match_rate": float(feature_nonnull.mean()) if len(joined) else 0.0,
        "pitcher_season_count": int(len(pitcher_season)),
        "pitcher_season_match_count": int(ps_covered.sum()),
        "pitcher_season_match_rate": float(ps_covered.mean()) if len(ps_covered) else 0.0,
        "attached_feature_columns": savant_feature_cols,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Join season-level Savant pitcher features onto pitcher game logs.")
    ap.add_argument("--game_logs", required=True, help="Path to pitcher_game_logs.csv/parquet")
    ap.add_argument("--savant", required=True, help="Path to pitcher_savant_features.csv/parquet")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--also_csv", action="store_true", help="Also write CSV")
    ap.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    print("Loading game logs...")
    games = load_any(args.game_logs)
    print("Loading Savant season features...")
    savant = load_any(args.savant)

    required_games = {"season", "pitcher_name"}
    required_savant = {"season", "pitcher_name"}
    missing_games = sorted(required_games - set(games.columns))
    missing_savant = sorted(required_savant - set(savant.columns))
    if missing_games:
        raise ValueError(f"Game logs missing required columns: {missing_games}")
    if missing_savant:
        raise ValueError(f"Savant table missing required columns: {missing_savant}")

    print("Normalizing join keys...")
    games = add_name_key(games, "pitcher_name")
    savant = add_name_key(savant, "pitcher_name")
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    savant["season"] = pd.to_numeric(savant["season"], errors="coerce").astype("Int64")

    savant = savant.dropna(subset=["season", NAME_KEY_COL]).copy()
    games = games.dropna(subset=["season", NAME_KEY_COL]).copy()

    # keep one savant row per season + normalized name
    savant = dedupe_savant(savant)

    # rename name column to preserve game-log pitcher_name as canonical display name
    if "pitcher_name" in savant.columns:
        savant = savant.rename(columns={"pitcher_name": "pitcher_name_savant"})

    merge_cols = ["season", NAME_KEY_COL]
    savant_feature_cols = [
        c for c in savant.columns
        if c not in set(merge_cols + ["pitcher_name_savant"]) and not c.startswith("source_")
    ]

    print("Joining Savant features to game logs by season + normalized pitcher name...")
    joined = games.merge(
        savant,
        how="left",
        on=merge_cols,
        suffixes=("", "_savantdup"),
    )

    dup_cols = [c for c in joined.columns if c.endswith("_savantdup")]
    if dup_cols:
        joined = joined.drop(columns=dup_cols)

    # helpful flags
    joined["has_savant_features"] = joined[savant_feature_cols].notna().any(axis=1) if savant_feature_cols else False
    if "pitcher_name_savant" in joined.columns:
        joined["pitcher_name_savant"] = joined["pitcher_name_savant"].astype("string")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_path, index=False)
    print(f"saved parquet: {out_path}")

    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        joined.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")

    summary = build_summary(joined, savant_feature_cols)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
