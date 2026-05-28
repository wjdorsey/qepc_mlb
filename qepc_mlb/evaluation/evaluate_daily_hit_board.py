#!/usr/bin/env python3
"""
Evaluate QEPC-MLB Daily Batter 1+ Hit Board results.

This checks a saved daily blend prediction file against actual batter results
from local processed batter game logs.

Example:

python qepc_mlb/evaluation/evaluate_daily_hit_board.py \
  --date 2025-09-28 \
  --top_n 25
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


SAFETY_VERSION = "evaluate_daily_batter_1plus_hit_board_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument(
        "--predictions",
        default=None,
        help="Optional all-predictions parquet. Defaults to blend output for date.",
    )
    p.add_argument(
        "--actuals",
        default="cache/mlb/processed/batter_game_logs.parquet",
        help="Processed batter game logs with actual hits.",
    )
    p.add_argument(
        "--out_dir",
        default="artifacts/mlb/evaluation/batter_1plus_hit_blend",
    )
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


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        return df

    if "date" in df.columns:
        raw = df["date"].astype("string").str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        fallback = pd.to_datetime(df["date"], errors="coerce")
        df["game_date"] = parsed.fillna(fallback).dt.date
        return df

    raise ValueError("Missing game_date/date column.")


def add_actual_hit_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "hit_1plus" in df.columns:
        df["actual_hit_1plus"] = pd.to_numeric(df["hit_1plus"], errors="coerce")
        return df

    hits_col = first_existing(df, ["hits", "b_h", "h"])
    if hits_col is None:
        raise ValueError("Could not find actual hits column: expected hit_1plus, hits, b_h, or h.")

    df["actual_hit_1plus"] = (pd.to_numeric(df[hits_col], errors="coerce").fillna(0) >= 1).astype(int)
    return df


def normalize_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace(".", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b[a-z]\b", " ", s)  # drop middle initials like Michael A Taylor
    return re.sub(r"\s+", " ", s).strip()


def build_actual_key_table(actuals: pd.DataFrame) -> pd.DataFrame:
    a = normalize_date_col(actuals)
    a = add_actual_hit_col(a)

    # Local Retrosheet-style files usually have batter_id.
    # MLB Stats API actuals usually have mlb_batter_id.
    if "batter_id" not in a.columns and "mlb_batter_id" in a.columns:
        a["batter_id"] = a["mlb_batter_id"]

    if "batter_id" in a.columns:
        a["batter_id"] = a["batter_id"].astype("string")

    if "mlb_batter_id" in a.columns:
        a["mlb_batter_id"] = a["mlb_batter_id"].astype("string")

    if "batter_name" in a.columns:
        a["batter_name_key"] = a["batter_name"].map(normalize_name)

    for c in ["game_id", "team", "opponent"]:
        if c in a.columns:
            a[c] = a[c].astype("string")

    keep = [
        c for c in [
            "game_id",
            "game_date",
            "batter_id",
            "mlb_batter_id",
            "team",
            "opponent",
            "actual_hit_1plus",
            "hits",
            "batter_name",
            "batter_name_key",
        ]
        if c in a.columns
    ]

    a = a[keep].copy()

    # Keep broad uniqueness here; the actual merge strategy is chosen later.
    possible_key_sets = [
        ["game_id", "batter_id"],
        ["game_date", "batter_id", "team", "opponent"],
        ["game_date", "mlb_batter_id", "team", "opponent"],
        ["game_date", "batter_name_key", "team", "opponent"],
        ["game_date", "batter_name_key", "team"],
    ]

    for key in possible_key_sets:
        if all(c in a.columns for c in key):
            return a.drop_duplicates(key, keep="first")

    return a.drop_duplicates(keep, keep="first")


def evaluate_topk(df: pd.DataFrame, k: int) -> dict:
    top = df.head(k).copy()
    return {
        f"top{k}_rows": int(len(top)),
        f"top{k}_hit_count": int(top["actual_hit_1plus"].sum()) if len(top) else 0,
        f"top{k}_hit_rate": float(top["actual_hit_1plus"].mean()) if len(top) else None,
    }


def main() -> None:
    args = parse_args()

    date_tag = args.date.replace("-", "")
    pred_path = (
        Path(args.predictions)
        if args.predictions
        else Path(f"artifacts/mlb/predictions/batter_1plus_hit_blend/batter_1plus_hit_blend_all_predictions_{date_tag}.parquet")
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = read_any(pred_path)
    actuals = build_actual_key_table(read_any(args.actuals))

    preds = normalize_date_col(preds)
    preds = preds[preds["game_date"].astype(str).eq(args.date)].copy()

    if preds.empty:
        raise SystemExit(f"No predictions found for date {args.date} in {pred_path}")

    if "batter_id" not in preds.columns:
        raise ValueError("Predictions missing batter_id.")

    preds["batter_id"] = preds["batter_id"].astype("string")

    if "mlb_batter_id" in preds.columns:
        preds["mlb_batter_id"] = preds["mlb_batter_id"].astype("string")

    if "batter_name" in preds.columns:
        preds["batter_name_key"] = preds["batter_name"].map(normalize_name)

    for c in ["game_id", "team", "opponent"]:
        if c in preds.columns:
            preds[c] = preds[c].astype("string")

    score_col = first_existing(preds, ["blend_prob", "pred_prob", "v3_prob"])
    if score_col is None:
        raise ValueError("Predictions missing blend_prob/pred_prob/v3_prob.")

    # Try multiple merge strategies.
    # Why: daily scoring may use MLB Stats API game ids, while historical actuals may use
    # Retrosheet-style game ids. If game_id fails, fall back to date/player/team matching.
    candidate_keys = []

    if "game_id" in preds.columns and "game_id" in actuals.columns:
        candidate_keys.append(["game_id", "batter_id"])

    if all(c in preds.columns for c in ["game_date", "batter_id", "team", "opponent"]) and all(c in actuals.columns for c in ["game_date", "batter_id", "team", "opponent"]):
        candidate_keys.append(["game_date", "batter_id", "team", "opponent"])

    if all(c in preds.columns for c in ["game_date", "mlb_batter_id", "team", "opponent"]) and all(c in actuals.columns for c in ["game_date", "mlb_batter_id", "team", "opponent"]):
        candidate_keys.append(["game_date", "mlb_batter_id", "team", "opponent"])

    if all(c in preds.columns for c in ["game_date", "batter_name_key", "team", "opponent"]) and all(c in actuals.columns for c in ["game_date", "batter_name_key", "team", "opponent"]):
        candidate_keys.append(["game_date", "batter_name_key", "team", "opponent"])

    if all(c in preds.columns for c in ["game_date", "batter_id", "team"]) and all(c in actuals.columns for c in ["game_date", "batter_id", "team"]):
        candidate_keys.append(["game_date", "batter_id", "team"])

    if all(c in preds.columns for c in ["game_date", "mlb_batter_id", "team"]) and all(c in actuals.columns for c in ["game_date", "mlb_batter_id", "team"]):
        candidate_keys.append(["game_date", "mlb_batter_id", "team"])

    if all(c in preds.columns for c in ["game_date", "batter_name_key", "team"]) and all(c in actuals.columns for c in ["game_date", "batter_name_key", "team"]):
        candidate_keys.append(["game_date", "batter_name_key", "team"])

    if all(c in preds.columns for c in ["game_date", "batter_id"]) and all(c in actuals.columns for c in ["game_date", "batter_id"]):
        candidate_keys.append(["game_date", "batter_id"])

    if all(c in preds.columns for c in ["game_date", "mlb_batter_id"]) and all(c in actuals.columns for c in ["game_date", "mlb_batter_id"]):
        candidate_keys.append(["game_date", "mlb_batter_id"])

    if not candidate_keys:
        raise ValueError("No usable merge keys found between predictions and actuals.")

    merge_attempts = []
    best = None

    for keys_try in candidate_keys:
        a_try = actuals.drop_duplicates(keys_try, keep="first").copy()

        merged_try = preds.merge(
            a_try,
            how="left",
            on=keys_try,
            suffixes=("", "_actual"),
        )

        missing_try = int(merged_try["actual_hit_1plus"].isna().sum())
        matched_try = int(len(merged_try) - missing_try)

        merge_attempts.append(
            {
                "keys": keys_try,
                "matched_actuals": matched_try,
                "missing_actuals": missing_try,
                "match_rate": float(matched_try / len(merged_try)) if len(merged_try) else 0.0,
            }
        )

        if best is None or missing_try < best["missing"]:
            best = {
                "keys": keys_try,
                "merged": merged_try,
                "missing": missing_try,
                "matched": matched_try,
            }

    keys = best["keys"]
    merged = best["merged"]
    missing_actuals = int(best["missing"])

    merged["actual_hit_1plus"] = pd.to_numeric(merged["actual_hit_1plus"], errors="coerce")

    print("\nMerge attempts:")
    for attempt in merge_attempts:
        print(attempt)

    ranked = merged.sort_values(score_col, ascending=False).copy()
    ranked["eval_rank"] = range(1, len(ranked) + 1)

    # Save missing actuals for debugging API/name/id mismatches.
    missing_debug = ranked[ranked["actual_hit_1plus"].isna()].copy()
    missing_debug_path = out_dir / f"missing_actuals_debug_{date_tag}.csv"
    if len(missing_debug):
        debug_cols = [
            "eval_rank",
            "game_date",
            "game_id",
            "batter_id",
            "mlb_batter_id",
            "batter_name",
            "batter_name_key",
            "team",
            "opponent",
            "lineup_slot",
            score_col,
        ]
        debug_cols = [c for c in debug_cols if c in missing_debug.columns]
        missing_debug[debug_cols].to_csv(missing_debug_path, index=False)

    top_board = ranked.head(args.top_n).copy()

    summary = {
        "safety_version": SAFETY_VERSION,
        "date": args.date,
        "predictions": str(pred_path),
        "actuals": args.actuals,
        "score_col": score_col,
        "merge_keys": keys,
        "merge_attempts": merge_attempts,
        "rows_predicted": int(len(preds)),
        "rows_after_merge": int(len(merged)),
        "missing_actuals": missing_actuals,
        "missing_actuals_debug": str(missing_debug_path) if missing_actuals else None,
        "top_n": args.top_n,
    }

    for k in [5, 10, 25, 50]:
        if k <= len(ranked):
            summary.update(evaluate_topk(ranked, k))

    out_csv = out_dir / f"batter_1plus_hit_blend_eval_top{args.top_n}_{date_tag}.csv"
    out_json = out_dir / f"batter_1plus_hit_blend_eval_summary_{date_tag}.json"

    display_cols = [
        "eval_rank",
        "batter_name",
        "team",
        "opponent",
        "lineup_slot",
        score_col,
        "v3_prob",
        "savant_prob",
        "blend_prob_base_80_20",
        "v3_rank_daily",
        "actual_hit_1plus",
        "hits",
    ]
    display_cols = [c for c in display_cols if c in top_board.columns]

    top_board[display_cols].to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    print("\nTop board result preview:")
    print(top_board[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
