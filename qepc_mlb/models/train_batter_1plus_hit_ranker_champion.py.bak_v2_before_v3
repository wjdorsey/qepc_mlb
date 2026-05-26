#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Ranker Champion Trainer.

Target:
    hit_1plus = 1 if batter records at least one hit, else 0.

Champion ranker:
    HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.0
    )

This is intended for the after-lineups-posted model:
    --starters_only --confirmed_lineups

In-sample training metrics are diagnostics only. Trust walk-forward metrics for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SAFETY_VERSION = "batter_1plus_hit_ranker_champion_train_v2_opp_starter_context_hgb_lr005_leaf15_l2001"
TARGET_COL = "hit_1plus"

SAME_GAME_OUTCOME_FEATURES = sorted({
    "at_bats", "b_ab", "b_cs", "b_d", "b_gdp", "b_h", "b_hbp", "b_hr", "b_k",
    "b_pa", "b_r", "b_rbi", "b_sb", "b_sf", "b_sh", "b_t", "b_w",
    "caught_stealing", "doubles", "grounded_into_double_play", "hit_1plus",
    "hit_by_pitch", "hits", "home_runs", "plate_appearances", "rbi", "runs",
    "sacrifice_flies", "sacrifice_hits", "stolen_bases", "strikeouts",
    "total_bases", "triples", "walks",
})

STARTER_ALIASES = [
    "is_starter",
    "is_starting_lineup",
    "is_starting_lineup_by_batting_file",
    "is_starting_lineup_teamstats",
    "starter_flag",
    "confirmed_starter",
]

NUMERIC_FEATURE_CANDIDATES = [
    "is_home",
    "env_temp",
    "env_windspeed",
    "has_env_context",

    "batter_games_prior",
    "games_prior",
    "batter_days_rest",

    "hit_1plus_lag1",
    "hit_1plus_roll3",
    "hit_1plus_roll5",
    "hit_1plus_roll10",
    "hit_1plus_roll20",

    "hits_lag1",
    "hits_roll3",
    "hits_roll5",
    "hits_roll10",
    "hits_roll20",

    "at_bats_lag1",
    "at_bats_roll3",
    "at_bats_roll5",
    "at_bats_roll10",
    "at_bats_roll20",

    "plate_appearances_lag1",
    "plate_appearances_roll3",
    "plate_appearances_roll5",
    "plate_appearances_roll10",
    "plate_appearances_roll20",

    "total_bases_lag1",
    "total_bases_roll3",
    "total_bases_roll5",
    "total_bases_roll10",
    "total_bases_roll20",

    "walks_lag1",
    "walks_roll3",
    "walks_roll5",
    "walks_roll10",
    "walks_roll20",

    "strikeouts_lag1",
    "strikeouts_roll3",
    "strikeouts_roll5",
    "strikeouts_roll10",
    "strikeouts_roll20",

    "home_runs_lag1",
    "home_runs_roll3",
    "home_runs_roll5",
    "home_runs_roll10",
    "home_runs_roll20",

    "batter_ba_roll5",
    "batter_k_rate_roll5",
    "batter_bb_rate_roll5",
    "batter_ba_roll10",
    "batter_k_rate_roll10",
    "batter_bb_rate_roll10",
    "batter_ba_roll20",
    "batter_k_rate_roll20",
    "batter_bb_rate_roll20",

    "hits_season_prior",
    "at_bats_season_prior",
    "plate_appearances_season_prior",
    "strikeouts_season_prior",
    "walks_season_prior",
    "batter_ba_season_prior",
    "batter_k_rate_season_prior",
    "batter_bb_rate_season_prior",

    "lineup_slot",
    "is_starting_lineup",
    "is_starting_lineup_by_batting_file",
    "is_starting_lineup_teamstats",
    "is_starter",
    "has_opp_starter_context",
    "opp_starter_games_prior",
    "opp_starter_days_rest",
    "opp_starter_strikeouts_lag1",
    "opp_starter_strikeouts_roll3",
    "opp_starter_strikeouts_roll5",
    "opp_starter_strikeouts_roll10",
    "opp_starter_batters_faced_lag1",
    "opp_starter_batters_faced_roll3",
    "opp_starter_batters_faced_roll5",
    "opp_starter_batters_faced_roll10",
    "opp_starter_hits_allowed_lag1",
    "opp_starter_hits_allowed_roll3",
    "opp_starter_hits_allowed_roll5",
    "opp_starter_hits_allowed_roll10",
    "opp_starter_walks_allowed_lag1",
    "opp_starter_walks_allowed_roll3",
    "opp_starter_walks_allowed_roll5",
    "opp_starter_walks_allowed_roll10",
    "opp_starter_runs_allowed_lag1",
    "opp_starter_runs_allowed_roll3",
    "opp_starter_runs_allowed_roll5",
    "opp_starter_runs_allowed_roll10",
    "opp_starter_home_runs_allowed_lag1",
    "opp_starter_home_runs_allowed_roll3",
    "opp_starter_home_runs_allowed_roll5",
    "opp_starter_home_runs_allowed_roll10",
    "opp_starter_outs_recorded_lag1",
    "opp_starter_outs_recorded_roll3",
    "opp_starter_outs_recorded_roll5",
    "opp_starter_outs_recorded_roll10",
    "opp_starter_innings_pitched_lag1",
    "opp_starter_innings_pitched_roll3",
    "opp_starter_innings_pitched_roll5",
    "opp_starter_innings_pitched_roll10",
    "opp_starter_k_per_bf_lag1",
    "opp_starter_k_per_bf_roll3",
    "opp_starter_k_per_bf_roll5",
    "opp_starter_k_per_bf_roll10",
    "opp_starter_hits_per_bf_lag1",
    "opp_starter_hits_per_bf_roll3",
    "opp_starter_hits_per_bf_roll5",
    "opp_starter_hits_per_bf_roll10",
    "opp_starter_bb_per_bf_lag1",
    "opp_starter_bb_per_bf_roll3",
    "opp_starter_bb_per_bf_roll5",
    "opp_starter_bb_per_bf_roll10",
    "opp_starter_hr_per_bf_lag1",
    "opp_starter_hr_per_bf_roll3",
    "opp_starter_hr_per_bf_roll5",
    "opp_starter_hr_per_bf_roll10",
    "opp_starter_er_per_ip_lag1",
    "opp_starter_er_per_ip_roll3",
    "opp_starter_er_per_ip_roll5",
    "opp_starter_er_per_ip_roll10",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "batter_hand",
    "team",
    "opponent",
    "home_away",
    "env_winddir",
    "env_daynight",
    "env_usedh",
    "opp_starter_throw",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QEPC-MLB Batter 1+ Hit Ranker Champion")
    p.add_argument("--input", required=True, help="Path to batter_game_logs parquet/csv")
    p.add_argument("--out_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_champion_v2")
    p.add_argument("--min_season", type=int, default=2022)
    p.add_argument("--max_season", type=int, default=2025)
    p.add_argument("--starters_only", action="store_true")
    p.add_argument("--confirmed_lineups", action="store_true")
    p.add_argument("--min_games", type=int, default=20)
    p.add_argument("--require_env_rows", action="store_true")
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def coerce_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).ne(0)
    text = s.astype("string").str.strip().str.lower()
    return text.isin(["1", "1.0", "true", "t", "yes", "y"])


def ensure_datetime_and_season(df: pd.DataFrame) -> pd.DataFrame:
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


def add_starter_alias(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    df = df.copy()
    starter_source = None

    for c in STARTER_ALIASES:
        if c in df.columns:
            starter_source = c
            break

    if starter_source is not None:
        df["is_starter"] = coerce_bool_series(df[starter_source])
    elif "lineup_slot" in df.columns:
        slot = pd.to_numeric(df["lineup_slot"], errors="coerce")
        df["is_starter"] = slot.between(1, 9)
        starter_source = "lineup_slot"
    else:
        df["is_starter"] = False

    return df, starter_source


def apply_filters(
    df: pd.DataFrame,
    *,
    min_season: Optional[int],
    max_season: Optional[int],
    starters_only: bool,
    confirmed_lineups: bool,
    min_games: int,
    require_env_rows: bool,
) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([0, 1])].copy()

    if min_season is not None:
        df = df[df["season"].astype(float) >= min_season].copy()
    if max_season is not None:
        df = df[df["season"].astype(float) <= max_season].copy()

    if starters_only:
        if "is_starter" not in df.columns:
            raise ValueError("--starters_only passed but starter flag unavailable.")
        df = df[df["is_starter"].fillna(False).astype(bool)].copy()

    if confirmed_lineups:
        if "lineup_slot" in df.columns:
            slot = pd.to_numeric(df["lineup_slot"], errors="coerce")
            df = df[slot.between(1, 9)].copy()
        elif "is_starter" in df.columns:
            df = df[df["is_starter"].fillna(False).astype(bool)].copy()

    if require_env_rows and "has_env_context" in df.columns:
        df = df[coerce_bool_series(df["has_env_context"])].copy()

    history_cols = [c for c in ["batter_games_prior", "games_prior"] if c in df.columns]
    if history_cols and min_games > 0:
        hist = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in history_cols}).max(axis=1)
        df = df[hist >= min_games].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    return df


def choose_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    forbidden = set(SAME_GAME_OUTCOME_FEATURES)

    numeric = []
    for c in NUMERIC_FEATURE_CANDIDATES:
        if c in df.columns and c not in forbidden:
            numeric.append(c)

    categorical = []
    for c in CATEGORICAL_FEATURE_CANDIDATES:
        if c in df.columns and c not in forbidden:
            categorical.append(c)

    numeric = list(dict.fromkeys(numeric))
    categorical = [c for c in dict.fromkeys(categorical) if c not in numeric]

    dropped_empty = []

    keep_numeric = []
    for c in numeric:
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().any():
            keep_numeric.append(c)
        else:
            dropped_empty.append(c)

    keep_categorical = []
    for c in categorical:
        ser = df[c].astype("string")
        if ser.notna().any():
            keep_categorical.append(c)
        else:
            dropped_empty.append(c)

    if not keep_numeric and not keep_categorical:
        raise ValueError("No usable features after filtering.")

    return keep_numeric, keep_categorical, dropped_empty


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def prepare_X(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    X = df[numeric_features + categorical_features].copy()
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_features:
        X[c] = X[c].astype("string").fillna("__MISSING__")
    return X


def classification_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    out["accuracy_50"] = float(accuracy_score(y, p >= 0.5))
    out["actual_rate"] = float(np.mean(y))
    out["avg_pred_prob"] = float(np.mean(p))
    out["prob_bias"] = float(np.mean(p) - np.mean(y))

    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = None

    try:
        out["avg_precision"] = float(average_precision_score(y, p))
    except Exception:
        out["avg_precision"] = None

    return out


def daily_topk_hit_rates(preds: pd.DataFrame, ks: Sequence[int] = (5, 10, 25, 50)) -> Dict[str, Any]:
    if "game_date" not in preds.columns or "pred_prob_1plus_hit" not in preds.columns:
        return {}

    work = preds.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    work = work.dropna(subset=["game_date", "pred_prob_1plus_hit", TARGET_COL]).copy()

    if work.empty:
        return {}

    work["game_day"] = work["game_date"].dt.date

    out: Dict[str, Any] = {}
    for k in ks:
        rows = []
        for _, g in work.groupby("game_day", sort=True):
            rows.append(g.sort_values("pred_prob_1plus_hit", ascending=False).head(k))

        if not rows:
            continue

        top = pd.concat(rows, ignore_index=True)
        out[f"top{k}_daily_hit_rate"] = float(top[TARGET_COL].mean())
        out[f"top{k}_daily_dates"] = int(top["game_day"].nunique())
        out[f"top{k}_daily_rows"] = int(len(top))

    return out


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame(Path(args.input))
    raw_rows = len(df)

    df = ensure_datetime_and_season(df)
    df, starter_source = add_starter_alias(df)

    leaky_found = [c for c in SAME_GAME_OUTCOME_FEATURES if c in df.columns and c != TARGET_COL]

    df_model = apply_filters(
        df,
        min_season=args.min_season,
        max_season=args.max_season,
        starters_only=args.starters_only,
        confirmed_lineups=args.confirmed_lineups,
        min_games=args.min_games,
        require_env_rows=args.require_env_rows,
    )

    numeric_features, categorical_features, dropped_empty = choose_features(df_model)

    X = prepare_X(df_model, numeric_features, categorical_features)
    y = df_model[TARGET_COL].astype(int).to_numpy()

    pipe = make_pipeline(numeric_features, categorical_features, args.random_state)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]

    preds = df_model.copy()
    preds["pred_prob_1plus_hit"] = p
    preds["game_day"] = preds["game_date"].dt.date
    preds["pred_rank_daily"] = preds.groupby("game_day")["pred_prob_1plus_hit"].rank(
        method="first",
        ascending=False,
    )

    metrics = classification_metrics(y, p)
    metrics.update(daily_topk_hit_rates(preds))

    model_path = out_dir / "model.joblib"
    metadata_path = out_dir / "metadata.json"
    feature_list_path = out_dir / "feature_list.json"
    predictions_path = out_dir / "training_predictions.parquet"
    top_board_path = out_dir / "training_top25_daily_board.parquet"

    joblib.dump(pipe, model_path)

    feature_list = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_empty_features": dropped_empty,
        "excluded_same_game_outcome_features": SAME_GAME_OUTCOME_FEATURES,
    }
    save_json(feature_list_path, feature_list)

    pred_cols = [
        c for c in [
            "game_id",
            "gid",
            "game_date",
            "season",
            "batter_id",
            "id",
            "batter_name",
            "team",
            "opponent",
            "home_away",
            "lineup_slot",
            TARGET_COL,
            "pred_prob_1plus_hit",
            "pred_rank_daily",
        ]
        if c in preds.columns
    ]

    preds[pred_cols].to_parquet(predictions_path, index=False)

    top25 = (
        preds.sort_values(["game_date", "pred_prob_1plus_hit"], ascending=[True, False])
        .groupby("game_day", group_keys=False)
        .head(25)
    )
    top25[pred_cols].to_parquet(top_board_path, index=False)

    csv_paths: Dict[str, str] = {}
    if args.also_csv:
        pred_csv = out_dir / "training_predictions.csv"
        board_csv = out_dir / "training_top25_daily_board.csv"
        preds[pred_cols].to_csv(pred_csv, index=False)
        top25[pred_cols].to_csv(board_csv, index=False)
        csv_paths["training_predictions_csv"] = str(pred_csv)
        csv_paths["training_top25_daily_board_csv"] = str(board_csv)

    batter_id_col = None
    for c in ["batter_id", "id"]:
        if c in df_model.columns:
            batter_id_col = c
            break

    metadata = {
        "safety_version": SAFETY_VERSION,
        "input": args.input,
        "raw_rows": int(raw_rows),
        "rows_after_filtering": int(len(df_model)),
        "seasons": [int(x) for x in sorted(df_model["season"].dropna().astype(int).unique().tolist())],
        "batters": int(df_model[batter_id_col].nunique()) if batter_id_col else None,
        "starter_source_column": starter_source,
        "confirmed_lineups": bool(args.confirmed_lineups),
        "starters_only": bool(args.starters_only),
        "min_games": int(args.min_games),
        "target": TARGET_COL,
        "target_rate": float(df_model[TARGET_COL].mean()),
        "champion_config": {
            "model": "HistGradientBoostingClassifier",
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.01,
            "random_state": args.random_state,
        },
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "dropped_empty_features": dropped_empty,
        "leaky_same_game_columns_present_but_excluded_from_features": leaky_found,
        "excluded_same_game_outcome_features": SAME_GAME_OUTCOME_FEATURES,
        "training_metrics_in_sample": metrics,
        "artifact_paths": {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "feature_list": str(feature_list_path),
            "training_predictions": str(predictions_path),
            "training_top25_daily_board": str(top_board_path),
            **csv_paths,
        },
        "note": "In-sample metrics are diagnostics only. Trust walk-forward ranker metrics for model selection.",
    }

    save_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
