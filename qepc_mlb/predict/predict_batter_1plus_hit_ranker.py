#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Ranker Predictor.

Loads the trained Batter 1+ Hit Ranker champion model and creates a ranked board:
    player most likely to record 1+ hit

Default behavior:
    - uses latest date available in the input table
    - filters to confirmed starters / lineup slots 1-9
    - requires at least --min_games prior games
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd


SAFETY_VERSION = "batter_1plus_hit_predict_v1"
TARGET_COL = "hit_1plus"

STARTER_ALIASES = [
    "is_starter",
    "is_starting_lineup",
    "is_starting_lineup_by_batting_file",
    "is_starting_lineup_teamstats",
    "starter_flag",
    "confirmed_starter",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict/rank batter 1+ hit probabilities")
    p.add_argument("--input", required=True)
    p.add_argument("--model_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_champion")
    p.add_argument("--out_dir", default="artifacts/mlb/predictions/batter_1plus_hit_ranker")
    p.add_argument("--date", default=None, help="YYYY-MM-DD. If omitted, latest date in input is used.")
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--min_games", type=int, default=20)
    p.add_argument("--include_non_starters", action="store_true")
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--save_all_predictions", action="store_true")
    return p.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).ne(0)
    text = s.astype("string").str.strip().str.lower()
    return text.isin(["1", "1.0", "true", "t", "yes", "y"])


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def add_starter_alias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in STARTER_ALIASES:
        if c in df.columns:
            df["is_starter"] = coerce_bool(df[c])
            return df

    if "lineup_slot" in df.columns:
        slot = pd.to_numeric(df["lineup_slot"], errors="coerce")
        df["is_starter"] = slot.between(1, 9)
    else:
        df["is_starter"] = False

    return df


def prepare_x(df: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> pd.DataFrame:
    numeric_data = {}
    categorical_data = {}

    for c in numeric_features:
        if c in df.columns:
            numeric_data[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            numeric_data[c] = pd.Series(np.nan, index=df.index)

    for c in categorical_features:
        if c in df.columns:
            categorical_data[c] = df[c].astype("string").fillna("__MISSING__")
        else:
            categorical_data[c] = pd.Series("__MISSING__", index=df.index, dtype="string")

    x = pd.concat(
        [
            pd.DataFrame(numeric_data, index=df.index),
            pd.DataFrame(categorical_data, index=df.index),
        ],
        axis=1,
    )

    return x[list(numeric_features) + list(categorical_features)].copy()


def output_columns(df: pd.DataFrame) -> List[str]:
    wanted = [
        "game_date",
        "season",
        "game_id",
        "gid",
        "batter_id",
        "id",
        "batter_name",
        "team",
        "opponent",
        "home_away",
        "lineup_slot",
        "batter_hand",
        "pred_rank_daily",
        "pred_prob_1plus_hit",
        "pred_prob_pct",
        TARGET_COL,
    ]
    return [c for c in wanted if c in df.columns]


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_dir / "model.joblib")
    feature_list = load_json(model_dir / "feature_list.json")
    metadata_path = model_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}

    numeric_features = feature_list["numeric_features"]
    categorical_features = feature_list["categorical_features"]

    df = load_frame(Path(args.input))
    raw_rows = len(df)

    df = ensure_date(df)
    df = add_starter_alias(df)
    df = df.dropna(subset=["game_date"]).copy()

    if args.date:
        score_date = pd.to_datetime(args.date).date()
    else:
        score_date = df["game_date"].max().date()

    score_df = df[df["game_date"].dt.date == score_date].copy()

    if not args.include_non_starters:
        if "lineup_slot" in score_df.columns:
            slot = pd.to_numeric(score_df["lineup_slot"], errors="coerce")
            score_df = score_df[slot.between(1, 9)].copy()
        else:
            score_df = score_df[score_df["is_starter"].fillna(False).astype(bool)].copy()

    history_cols = [c for c in ["batter_games_prior", "games_prior"] if c in score_df.columns]
    if history_cols and args.min_games > 0:
        hist = pd.DataFrame({c: pd.to_numeric(score_df[c], errors="coerce") for c in history_cols}).max(axis=1)
        score_df = score_df[hist >= args.min_games].copy()

    if score_df.empty:
        raise ValueError("No rows left to score after filtering.")

    x = prepare_x(score_df, numeric_features, categorical_features)
    probs = model.predict_proba(x)[:, 1]

    score_df["pred_prob_1plus_hit"] = probs
    score_df["pred_prob_pct"] = (score_df["pred_prob_1plus_hit"] * 100).round(1)
    score_df["pred_rank_daily"] = score_df["pred_prob_1plus_hit"].rank(method="first", ascending=False)

    score_df = score_df.sort_values("pred_prob_1plus_hit", ascending=False).copy()
    top_board = score_df.head(args.top_n).copy()

    date_tag = str(score_date).replace("-", "")
    board_path = out_dir / f"batter_1plus_hit_top{args.top_n}_{date_tag}.parquet"
    summary_path = out_dir / f"batter_1plus_hit_prediction_summary_{date_tag}.json"

    cols = output_columns(top_board)
    top_board[cols].to_parquet(board_path, index=False)

    artifact_paths = {
        "top_board": str(board_path),
    }

    if args.also_csv:
        board_csv = board_path.with_suffix(".csv")
        top_board[cols].to_csv(board_csv, index=False)
        artifact_paths["top_board_csv"] = str(board_csv)

    if args.save_all_predictions:
        all_path = out_dir / f"batter_1plus_hit_all_predictions_{date_tag}.parquet"
        all_cols = output_columns(score_df)
        score_df[all_cols].to_parquet(all_path, index=False)
        artifact_paths["all_predictions"] = str(all_path)

        if args.also_csv:
            all_csv = all_path.with_suffix(".csv")
            score_df[all_cols].to_csv(all_csv, index=False)
            artifact_paths["all_predictions_csv"] = str(all_csv)

    summary = {
        "safety_version": SAFETY_VERSION,
        "model_dir": str(model_dir),
        "model_metadata_safety_version": metadata.get("safety_version"),
        "input": args.input,
        "raw_rows": int(raw_rows),
        "score_date": str(score_date),
        "rows_scored": int(len(score_df)),
        "top_n": int(args.top_n),
        "min_games": int(args.min_games),
        "include_non_starters": bool(args.include_non_starters),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "missing_numeric_features_in_input": [c for c in numeric_features if c not in df.columns],
        "missing_categorical_features_in_input": [c for c in categorical_features if c not in df.columns],
        "avg_pred_prob": float(score_df["pred_prob_1plus_hit"].mean()),
        "top_board_avg_pred_prob": float(top_board["pred_prob_1plus_hit"].mean()),
        "top_board_max_pred_prob": float(top_board["pred_prob_1plus_hit"].max()),
        "artifact_paths": artifact_paths,
        "note": "For true future/live use, input must contain future or confirmed-lineup rows with the same feature schema.",
    }

    if TARGET_COL in top_board.columns:
        top_target = pd.to_numeric(top_board[TARGET_COL], errors="coerce")
        all_target = pd.to_numeric(score_df[TARGET_COL], errors="coerce")
        if top_target.notna().any():
            summary["top_board_actual_hit_rate"] = float(top_target.mean())
        if all_target.notna().any():
            summary["all_scored_actual_hit_rate"] = float(all_target.mean())

    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    artifact_paths["summary"] = str(summary_path)

    print(json.dumps(summary, indent=2, default=str))
    print("\nTop board preview:")
    preview_cols = [c for c in ["pred_rank_daily", "batter_name", "team", "opponent", "lineup_slot", "pred_prob_pct", TARGET_COL] if c in top_board.columns]
    print(top_board[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
