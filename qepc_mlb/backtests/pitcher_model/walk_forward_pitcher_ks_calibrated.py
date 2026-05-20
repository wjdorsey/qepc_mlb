#!/usr/bin/env python3
"""
QEPC-MLB walk-forward pitcher strikeout calibration test.

Purpose
-------
Compare the current no-leak champion pitcher-K model against a simple
walk-forward affine-calibrated version:

    actual_K ~= intercept + slope * raw_prediction

Important safety rule:
- The calibration fit only uses rows that occur BEFORE the test window.
- Same-game lineup outcome columns are forcibly excluded.

Typical run:

python qepc_mlb/backtests/walk_forward_pitcher_ks_calibrated.py \
  --input cache/mlb/processed/pitcher_game_logs_enriched_opp_env_lineup.parquet \
  --min_season 2024 \
  --max_season 2025 \
  --starters_only \
  --min_games 5 \
  --require_savant_rows \
  --require_lineup_rows \
  --savant_feature_set stable
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SAFETY_VERSION = "pitcher_ks_calibrated_noleak_v1"
TARGET_COL = "strikeouts"
DATE_COL = "game_date"
PERIOD_COL = "wf_period"

# These columns describe same-game batting outcomes and must never be model inputs.
LEAKY_LINEUP_COLS = [
    "lineup_batting_ks",
    "lineup_batting_pa",
]

# Stable Savant features that existed consistently enough in the first 2024-2025 tests.
STABLE_SAVANT_FEATURES = [
    "savant_pa",
    "savant_bip",
    "savant_ba_allowed",
    "savant_xba_allowed",
    "savant_xba_minus_ba",
    "savant_slg_allowed",
    "savant_xslg_allowed",
    "savant_xslg_minus_slg",
    "savant_woba_allowed",
    "savant_xwoba_allowed",
    "savant_xwoba_minus_woba",
    "savant_era",
    "savant_xera",
    "savant_era_minus_xera",
]

# Optional wider Savant feature set. Useful for experiments, but the stable set is safer.
ALL_SAVANT_PREFIX = "savant_"

OPP_FEATURES = [
    "opp_team_games_prior",
    "opp_team_days_since_game",
    "opp_team_ks_lag1",
    "opp_team_ks_roll3",
    "opp_team_ks_std3",
    "opp_team_ks_roll5",
    "opp_team_ks_std5",
    "opp_team_ks_roll10",
    "opp_team_ks_std10",
    "opp_team_ks_expanding_mean",
    "opp_team_starter_ks_lag1",
    "opp_team_starter_ks_roll3",
    "opp_team_starter_ks_roll5",
    "opp_team_starter_ks_roll10",
    "opp_team_k_per_bf_lag1",
    "opp_team_k_per_bf_roll3",
    "opp_team_k_per_bf_roll5",
    "opp_team_k_per_bf_roll10",
]

ENV_NUMERIC_FEATURES = [
    "env_temp",
    "env_temp_sq",
    "env_windspeed",
    "env_windspeed_sq",
    "env_wind_out_effect",
    "env_wind_in_effect",
    "env_wind_cross_effect",
    "env_temp_night_interaction",
]

ENV_CATEGORICAL_FEATURES = [
    "env_park_id",
    "env_daynight",
    "env_winddir",
    "env_usedh",
    "env_is_night",
    "env_is_day",
    "env_has_dh",
    "env_no_dh",
    "env_wind_out",
    "env_wind_in",
    "env_wind_cross",
    "env_wind_calm",
    "env_temp_missing",
    "env_windspeed_missing",
    "env_is_home",
    "has_env_context",
]

SAFE_LINEUP_FEATURES = [
    "lineup_team_ks_lag1",
    "lineup_team_ks_roll3",
    "lineup_team_ks_roll5",
    "lineup_team_ks_roll10",
    "lineup_team_ks_expanding_mean",
    "lineup_team_ks_std5",
    "lineup_team_k_per_pa_lag1",
    "lineup_team_k_per_pa_roll5",
    "lineup_known_count",
    "lineup_known_count_lag1",
    "lineup_known_count_roll5",
    "lineup_returning_starters_prev",
    "lineup_returning_starters_roll5",
    "has_lineup_context",
]

BASE_NUMERIC_FEATURES = [
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "days_rest",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
    "games_prior",
    "target_lag1",
    "k_roll_mean_3",
    "k_roll_std_3",
    "k_roll_mean_5",
    "k_roll_std_5",
    "k_roll_mean_10",
    "k_roll_std_10",
    "batters_faced_lag1",
    "batters_faced_roll5",
    "innings_pitched_lag1",
    "innings_pitched_roll5",
    "outs_recorded_lag1",
    "outs_recorded_roll5",
    "days_rest_lag1",
    "days_rest_roll5",
    "walks_lag1",
    "walks_roll5",
    "hits_allowed_lag1",
    "hits_allowed_roll5",
    "earned_runs_lag1",
    "earned_runs_roll5",
    "runs_allowed_lag1",
    "runs_allowed_roll5",
    "k_per_bf_lag1",
    "k_per_bf_roll5",
]

BASE_CATEGORICAL_FEATURES = [
    "pitcher_hand",
    "team",
    "opponent",
    "home_away",
]


@dataclass
class WindowResult:
    period: str
    train_rows: int
    fit_rows: int
    calibration_rows: int
    test_rows: int
    raw_rmse: float
    raw_mae: float
    raw_bias: float
    calibrated_rmse: float
    calibrated_mae: float
    calibrated_bias: float
    calibration_slope: float
    calibration_intercept: float
    calibration_method: str
    feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    dropped_window_empty_count: int
    dropped_window_empty_features: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward affine calibration for QEPC-MLB pitcher Ks.")
    parser.add_argument("--input", required=True, help="Input parquet/csv with enriched pitcher game logs.")
    parser.add_argument("--out_dir", default="artifacts/mlb/backtests/pitcher_strikeouts_calibrated")
    parser.add_argument("--min_season", type=int, default=2024)
    parser.add_argument("--max_season", type=int, default=2025)
    parser.add_argument("--starters_only", action="store_true")
    parser.add_argument("--min_games", type=int, default=5, help="Minimum prior games for a row to be modelable.")
    parser.add_argument("--min_train_rows", type=int, default=500)
    parser.add_argument("--period", choices=["M", "MS", "W"], default="M", help="Walk-forward test period frequency.")
    parser.add_argument("--require_savant_rows", action="store_true")
    parser.add_argument("--require_lineup_rows", action="store_true")
    parser.add_argument("--require_env_rows", action="store_true")
    parser.add_argument("--savant_feature_set", choices=["stable", "all", "none"], default="stable")
    parser.add_argument("--model", choices=["rf", "extra_trees", "ridge"], default="rf")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--calibration_method", choices=["holdout", "insample", "none"], default="holdout")
    parser.add_argument("--calibration_holdout_frac", type=float, default=0.20)
    parser.add_argument("--min_calibration_rows", type=int, default=200)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=18.0)
    parser.add_argument("--save_csv_predictions", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path}")


def as_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    text = series.astype("string").str.strip().str.lower()
    out.loc[out.isna() & text.isin(["true", "t", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & text.isin(["false", "f", "no", "n", "reliever", "rp"])] = False
    return out.fillna(False).astype(bool)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Defensive dedupe: duplicated labels cause ColumnTransformer pain later.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if DATE_COL not in df.columns:
        raise ValueError(f"Input must contain {DATE_COL!r}.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Input must contain {TARGET_COL!r}.")
    if "pitcher_id" not in df.columns:
        raise ValueError("Input must contain 'pitcher_id'.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df[df[DATE_COL].notna()].copy()
    df["season"] = df[DATE_COL].dt.year.astype(int)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()

    if "is_starter" in df.columns:
        df["is_starter"] = as_bool(df["is_starter"])
    else:
        df["is_starter"] = False

    for col in LEAKY_LINEUP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df.sort_values(["pitcher_id", DATE_COL]).reset_index(drop=True)


def compute_days_rest_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "days_rest" not in df.columns or df["days_rest"].isna().all():
        df["days_rest"] = df.groupby("pitcher_id", sort=False)[DATE_COL].diff().dt.days
    return df


def add_pitcher_rolling_features(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    df = compute_days_rest_if_needed(df.copy())
    df = df.sort_values(["pitcher_id", DATE_COL]).reset_index(drop=True)

    groups = df.groupby("pitcher_id", group_keys=False, sort=False)
    df["games_prior"] = groups.cumcount()
    df["target_lag1"] = groups[TARGET_COL].shift(1)

    iterator = [3, 5, 10]
    if tqdm is not None and show_progress:
        iterator = tqdm(iterator, desc="Rolling K windows", leave=False)
    for window in iterator:
        shifted = groups[TARGET_COL].shift(1)
        df[f"k_roll_mean_{window}"] = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"k_roll_std_{window}"] = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)

    workload_cols = [
        "batters_faced",
        "innings_pitched",
        "outs_recorded",
        "days_rest",
        "walks",
        "hits_allowed",
        "earned_runs",
        "runs_allowed",
    ]
    for col in workload_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        shifted = groups[col].shift(1)
        df[f"{col}_lag1"] = shifted
        df[f"{col}_roll5"] = shifted.groupby(df["pitcher_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    if "batters_faced" in df.columns:
        bf = pd.to_numeric(df["batters_faced"], errors="coerce").replace(0, np.nan)
        df["k_per_bf"] = pd.to_numeric(df[TARGET_COL], errors="coerce") / bf
        shifted_rate = groups["k_per_bf"].shift(1)
        df["k_per_bf_lag1"] = shifted_rate
        df["k_per_bf_roll5"] = shifted_rate.groupby(df["pitcher_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    return df


def existing(cols: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def unique_preserve(cols: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def select_features(df: pd.DataFrame, savant_feature_set: str) -> tuple[list[str], list[str], dict]:
    numeric = []
    categorical = []

    numeric += existing(BASE_NUMERIC_FEATURES, df)
    categorical += existing(BASE_CATEGORICAL_FEATURES, df)

    if savant_feature_set == "stable":
        numeric += existing(STABLE_SAVANT_FEATURES, df)
    elif savant_feature_set == "all":
        numeric += [c for c in df.columns if c.startswith(ALL_SAVANT_PREFIX) and c not in {"savant_pitcher_name"}]

    numeric += existing(OPP_FEATURES, df)
    numeric += existing(ENV_NUMERIC_FEATURES, df)
    categorical += existing(ENV_CATEGORICAL_FEATURES, df)

    # Important: safe lineup features only. Same-game lineup_batting_* columns are excluded globally.
    for c in SAFE_LINEUP_FEATURES:
        if c not in df.columns:
            continue
        if c.startswith("has_") or df[c].dtype == "bool" or str(df[c].dtype) == "boolean":
            categorical.append(c)
        else:
            numeric.append(c)

    # Keep categoricals out of numeric list if a column was accidentally listed in both.
    categorical = unique_preserve(categorical)
    numeric = unique_preserve([c for c in numeric if c not in set(categorical)])

    meta = {
        "candidate_numeric_features": numeric,
        "candidate_categorical_features": categorical,
        "candidate_savant_features": [c for c in numeric if c.startswith("savant_")],
        "candidate_opp_features": [c for c in numeric if c.startswith("opp_")],
        "candidate_env_numeric_features": [c for c in numeric if c.startswith("env_")],
        "candidate_env_categorical_features": [c for c in categorical if c.startswith("env_") or c == "has_env_context"],
        "candidate_lineup_features": [c for c in numeric + categorical if c.startswith("lineup_") or c == "has_lineup_context"],
    }
    return numeric, categorical, meta


def clean_feature_frame(X: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    X = X.loc[:, unique_preserve(list(numeric_cols) + list(categorical_cols))].copy()
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X


def drop_empty_features(train_df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    keep_num = []
    keep_cat = []
    dropped = []
    for c in numeric_cols:
        if c not in train_df.columns or pd.to_numeric(train_df[c], errors="coerce").notna().sum() == 0:
            dropped.append(c)
        else:
            keep_num.append(c)
    for c in categorical_cols:
        if c not in train_df.columns or train_df[c].astype("string").notna().sum() == 0:
            dropped.append(c)
        else:
            keep_cat.append(c)
    return keep_num, keep_cat, dropped


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(model_name: str, random_state: int, n_estimators: int):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        )
    if model_name == "ridge":
        return Ridge(alpha=5.0, random_state=random_state)
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(numeric_cols: Sequence[str], categorical_cols: Sequence[str], model_name: str, random_state: int, n_estimators: int) -> Pipeline:
    transformers = []
    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "ridge":
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), list(numeric_cols)))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]),
            list(categorical_cols),
        ))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    model = make_model(model_name, random_state=random_state, n_estimators=n_estimators)
    return Pipeline([("preprocess", pre), ("model", model)])


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def score_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "bias": float(np.mean(y_pred_arr - y_true_arr)),
    }


def clip_preds(pred: np.ndarray, clip_min: float | None, clip_max: float | None) -> np.ndarray:
    out = np.asarray(pred, dtype=float)
    if clip_min is not None:
        out = np.maximum(out, clip_min)
    if clip_max is not None:
        out = np.minimum(out, clip_max)
    return out


def split_fit_calibration(train: pd.DataFrame, holdout_frac: float, min_cal_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.sort_values(DATE_COL).copy()
    n = len(train)
    cal_n = max(min_cal_rows, int(round(n * holdout_frac)))
    cal_n = min(cal_n, max(0, n // 2))  # Keep at least half for fitting.
    if cal_n < min_cal_rows or n - cal_n < min_cal_rows:
        return train.iloc[0:0].copy(), train.iloc[0:0].copy()
    return train.iloc[:-cal_n].copy(), train.iloc[-cal_n:].copy()


def fit_affine_calibrator(raw_cal_pred: np.ndarray, y_cal: pd.Series | np.ndarray) -> tuple[float, float]:
    raw_cal_pred = np.asarray(raw_cal_pred, dtype=float).reshape(-1, 1)
    y_cal = np.asarray(y_cal, dtype=float)
    if len(y_cal) < 2 or np.nanstd(raw_cal_pred) < 1e-8:
        return 1.0, 0.0
    lr = LinearRegression()
    lr.fit(raw_cal_pred, y_cal)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return 1.0, 0.0
    return slope, intercept


def apply_affine(pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return intercept + slope * np.asarray(pred, dtype=float)


def prepare_modelable_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = normalize_input(df)
    df = add_pitcher_rolling_features(df, show_progress=not args.no_progress)

    df = df[(df["season"] >= args.min_season) & (df["season"] <= args.max_season)].copy()
    if args.starters_only:
        df = df[df["is_starter"]].copy()
    if "games_prior" in df.columns:
        df = df[df["games_prior"] >= args.min_games].copy()
    if args.require_savant_rows and "has_savant_features" in df.columns:
        df = df[as_bool(df["has_savant_features"])].copy()
    if args.require_lineup_rows and "has_lineup_context" in df.columns:
        df = df[as_bool(df["has_lineup_context"])].copy()
    if args.require_env_rows and "has_env_context" in df.columns:
        df = df[as_bool(df["has_env_context"])].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering. Relax filters or check input table.")

    if args.period == "M":
        df[PERIOD_COL] = df[DATE_COL].dt.to_period("M").astype(str)
    elif args.period == "MS":
        df[PERIOD_COL] = df[DATE_COL].dt.to_period("M").dt.start_time.astype(str)
    elif args.period == "W":
        df[PERIOD_COL] = df[DATE_COL].dt.to_period("W").astype(str)
    else:
        raise ValueError(args.period)

    return df.sort_values(DATE_COL).reset_index(drop=True)


def run_walk_forward(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    numeric_candidates, categorical_candidates, feature_meta = select_features(df, args.savant_feature_set)
    periods = sorted(df[PERIOD_COL].dropna().unique())
    iterator = periods
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(periods, desc="Walk-forward calibrated", leave=True)

    window_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    all_dropped = []

    for period in iterator:
        test = df[df[PERIOD_COL] == period].copy()
        if test.empty:
            continue
        train = df[df[DATE_COL] < test[DATE_COL].min()].copy()
        if len(train) < args.min_train_rows:
            continue

        numeric_cols, categorical_cols, dropped = drop_empty_features(train, numeric_candidates, categorical_candidates)
        all_dropped.extend(dropped)
        if not numeric_cols and not categorical_cols:
            warnings.warn(f"Skipping {period}: no usable features.")
            continue

        feature_cols = unique_preserve(numeric_cols + categorical_cols)
        X_train = clean_feature_frame(train, numeric_cols, categorical_cols)
        X_test = clean_feature_frame(test, numeric_cols, categorical_cols)
        y_train = train[TARGET_COL].astype(float)
        y_test = test[TARGET_COL].astype(float)

        # Raw champion model: fit on all prior data and score this future period.
        raw_pipe = make_pipeline(numeric_cols, categorical_cols, args.model, args.random_state, args.n_estimators)
        raw_pipe.fit(X_train, y_train)
        raw_pred = clip_preds(raw_pipe.predict(X_test), args.clip_min, args.clip_max)

        slope, intercept = 1.0, 0.0
        cal_method_used = "none"

        if args.calibration_method == "insample":
            train_pred = clip_preds(raw_pipe.predict(X_train), args.clip_min, args.clip_max)
            slope, intercept = fit_affine_calibrator(train_pred, y_train)
            cal_method_used = "insample"
        elif args.calibration_method == "holdout":
            fit_train, cal_train = split_fit_calibration(train, args.calibration_holdout_frac, args.min_calibration_rows)
            if len(fit_train) > 0 and len(cal_train) > 0:
                X_fit = clean_feature_frame(fit_train, numeric_cols, categorical_cols)
                y_fit = fit_train[TARGET_COL].astype(float)
                X_cal = clean_feature_frame(cal_train, numeric_cols, categorical_cols)
                y_cal = cal_train[TARGET_COL].astype(float)

                cal_pipe = make_pipeline(numeric_cols, categorical_cols, args.model, args.random_state, args.n_estimators)
                cal_pipe.fit(X_fit, y_fit)
                cal_pred = clip_preds(cal_pipe.predict(X_cal), args.clip_min, args.clip_max)
                slope, intercept = fit_affine_calibrator(cal_pred, y_cal)
                cal_method_used = "holdout"
            else:
                cal_method_used = "fallback_identity"
        elif args.calibration_method == "none":
            cal_method_used = "none"
        else:
            raise ValueError(args.calibration_method)

        calibrated_pred = clip_preds(apply_affine(raw_pred, slope, intercept), args.clip_min, args.clip_max)

        raw_scores = score_predictions(y_test, raw_pred)
        cal_scores = score_predictions(y_test, calibrated_pred)

        window_rows.append({
            "period": period,
            "train_rows": int(len(train)),
            "fit_rows": int(len(train) if cal_method_used != "holdout" else max(len(train) - int(round(len(train) * args.calibration_holdout_frac)), 0)),
            "calibration_rows": int(0 if cal_method_used not in {"holdout", "insample"} else (len(train) if cal_method_used == "insample" else max(int(round(len(train) * args.calibration_holdout_frac)), args.min_calibration_rows))),
            "test_rows": int(len(test)),
            "raw_rmse": raw_scores["rmse"],
            "raw_mae": raw_scores["mae"],
            "raw_bias": raw_scores["bias"],
            "calibrated_rmse": cal_scores["rmse"],
            "calibrated_mae": cal_scores["mae"],
            "calibrated_bias": cal_scores["bias"],
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibration_method": cal_method_used,
            "feature_count": int(len(feature_cols)),
            "numeric_feature_count": int(len(numeric_cols)),
            "categorical_feature_count": int(len(categorical_cols)),
            "dropped_window_empty_count": int(len(dropped)),
            "dropped_window_empty_features": ",".join(dropped),
        })

        pred_out_cols = [
            c for c in [
                "game_id", DATE_COL, PERIOD_COL, "season", "pitcher_id", "pitcher_name", "team", "opponent", "pitcher_hand",
                TARGET_COL,
            ] if c in test.columns
        ]
        pf = test[pred_out_cols].copy()
        pf["raw_pred"] = raw_pred
        pf["calibrated_pred"] = calibrated_pred
        pf["raw_error"] = pf["raw_pred"] - pf[TARGET_COL].astype(float)
        pf["calibrated_error"] = pf["calibrated_pred"] - pf[TARGET_COL].astype(float)
        pf["calibration_slope"] = slope
        pf["calibration_intercept"] = intercept
        pf["calibration_method"] = cal_method_used
        pred_frames.append(pf)

    wm = pd.DataFrame(window_rows)
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if wm.empty or preds.empty:
        raise ValueError("No walk-forward windows were scored. Lower --min_train_rows or inspect filters.")

    raw_overall = score_predictions(preds[TARGET_COL], preds["raw_pred"])
    cal_overall = score_predictions(preds[TARGET_COL], preds["calibrated_pred"])

    weighted_raw_mae = float(np.average(wm["raw_mae"], weights=wm["test_rows"]))
    weighted_cal_mae = float(np.average(wm["calibrated_mae"], weights=wm["test_rows"]))
    weighted_raw_rmse = float(np.average(wm["raw_rmse"], weights=wm["test_rows"]))
    weighted_cal_rmse = float(np.average(wm["calibrated_rmse"], weights=wm["test_rows"]))
    weighted_raw_bias = float(np.average(wm["raw_bias"], weights=wm["test_rows"]))
    weighted_cal_bias = float(np.average(wm["calibrated_bias"], weights=wm["test_rows"]))

    summary = {
        "safety_version": SAFETY_VERSION,
        "model": args.model,
        "calibration_method": args.calibration_method,
        "rows_after_filtering": int(len(df)),
        "rows_scored": int(len(preds)),
        "windows": int(len(wm)),
        "seasons": sorted([int(x) for x in df["season"].dropna().unique()]),
        "globally_excluded_leaky_lineup_features": LEAKY_LINEUP_COLS,
        "feature_candidates": feature_meta,
        "avg_feature_count_per_window": float(wm["feature_count"].mean()),
        "avg_dropped_window_empty_count": float(wm["dropped_window_empty_count"].mean()),
        "raw": {
            "overall_rmse": raw_overall["rmse"],
            "overall_mae": raw_overall["mae"],
            "overall_bias": raw_overall["bias"],
            "weighted_window_mae": weighted_raw_mae,
            "weighted_window_rmse": weighted_raw_rmse,
            "weighted_window_bias": weighted_raw_bias,
        },
        "calibrated": {
            "overall_rmse": cal_overall["rmse"],
            "overall_mae": cal_overall["mae"],
            "overall_bias": cal_overall["bias"],
            "weighted_window_mae": weighted_cal_mae,
            "weighted_window_rmse": weighted_cal_rmse,
            "weighted_window_bias": weighted_cal_bias,
            "avg_slope": float(wm["calibration_slope"].mean()),
            "avg_intercept": float(wm["calibration_intercept"].mean()),
        },
        "comparison": {
            "overall_rmse_delta_calibrated_minus_raw": cal_overall["rmse"] - raw_overall["rmse"],
            "overall_mae_delta_calibrated_minus_raw": cal_overall["mae"] - raw_overall["mae"],
            "overall_bias_delta_calibrated_minus_raw": cal_overall["bias"] - raw_overall["bias"],
            "weighted_window_rmse_delta_calibrated_minus_raw": weighted_cal_rmse - weighted_raw_rmse,
            "weighted_window_mae_delta_calibrated_minus_raw": weighted_cal_mae - weighted_raw_mae,
            "weighted_window_bias_delta_calibrated_minus_raw": weighted_cal_bias - weighted_raw_bias,
        },
    }
    return wm, preds, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input: {args.input}")
    df = read_any(args.input)
    print(f"input rows: {len(df):,}")

    df = prepare_modelable_rows(df, args)
    print(f"rows after filtering/modelable prep: {len(df):,}")

    wm, preds, summary = run_walk_forward(df, args)

    window_path = out_dir / "window_metrics.csv"
    pred_path = out_dir / "predictions.parquet"
    summary_path = out_dir / "summary.json"

    wm.to_csv(window_path, index=False)
    preds.to_parquet(pred_path, index=False)
    if args.save_csv_predictions:
        preds.to_csv(out_dir / "predictions.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved window metrics: {window_path}")
    print(f"saved predictions: {pred_path}")
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
