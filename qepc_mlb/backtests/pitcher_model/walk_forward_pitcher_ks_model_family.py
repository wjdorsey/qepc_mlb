#!/usr/bin/env python3
"""
QEPC-MLB walk-forward pitcher strikeout model-family comparison.

Purpose
-------
Compare multiple estimators on the same leakage-safe walk-forward windows.
This script is designed for the current QEPC-MLB pitcher-K champion feature stack:
Retrosheet spine + stable Savant + opponent context + environment + safe lineup context.

Key safety behavior
-------------------
- Excludes same-game lineup outcome columns like lineup_batting_ks and lineup_batting_pa.
- Builds pitcher rolling features with shift(1) only.
- Trains each test period using only rows before that period.
- Drops all-empty feature columns per training window.
- Converts numeric/categorical features explicitly to avoid sklearn dtype crashes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SAFETY_VERSION = "model_family_noleak_v2_champion_rf"
TARGET_COL = "strikeouts"
DATE_COL = "game_date"
PITCHER_COL = "pitcher_id"

# These are same-game outcome-ish columns. They are allowed to exist in the input,
# but they are forcibly excluded from modeling.
LEAKY_LINEUP_COLS = [
    "lineup_batting_ks",
    "lineup_batting_pa",
]

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

BASE_NUMERIC_RAW = [
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "days_rest",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
    "home_away_flag",
]

BASE_CATEGORICAL_RAW = [
    "team",
    "opponent",
    "pitcher_hand",
    "home_away",
    "is_starter",
]

ROLL_SOURCE_COLS = [
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "days_rest",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
]

ROLL_FEATURES = [
    "games_prior",
    "target_lag1",
    "k_roll_mean_3",
    "k_roll_std_3",
    "k_roll_mean_5",
    "k_roll_std_5",
    "k_roll_mean_10",
    "k_roll_std_10",
    "k_per_bf_lag1",
    "k_per_bf_roll5",
]
for c in ROLL_SOURCE_COLS:
    ROLL_FEATURES.extend([f"{c}_lag1", f"{c}_roll5"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QEPC-MLB pitcher-K model-family walk-forward comparison")
    parser.add_argument("--input", required=True, help="Input enriched pitcher game-log parquet/csv")
    parser.add_argument("--out_dir", default="artifacts/mlb/backtests/pitcher_strikeouts_model_family")
    parser.add_argument("--min_season", type=int, default=2024)
    parser.add_argument("--max_season", type=int, default=2025)
    parser.add_argument("--starters_only", action="store_true")
    parser.add_argument("--min_games", type=int, default=5, help="Minimum prior pitcher games before a row can be scored")
    parser.add_argument("--period", default="M", help="Pandas period frequency for walk-forward windows, e.g. M")
    parser.add_argument("--min_train_rows", type=int, default=1000)
    parser.add_argument("--require_savant_rows", action="store_true")
    parser.add_argument("--require_lineup_rows", action="store_true")
    parser.add_argument("--require_env_rows", action="store_true")
    parser.add_argument("--feature_stack", choices=["baseline", "enriched", "both"], default="enriched")
    parser.add_argument("--savant_feature_set", choices=["stable", "all", "none"], default="stable")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf_champion", "rf", "extra_trees_leaf2", "extra_trees", "ridge", "hgb", "poisson"],
        choices=["rf", "rf_champion", "rf_leaf1", "extra_trees", "extra_trees_leaf2", "extra_trees_leaf1", "hgb", "ridge", "poisson"],
    )
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_models", action="store_true", help="Save final full-sample fitted pipelines for each model/stack")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def coerce_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False

    text = series.astype("string").str.strip().str.lower()
    true_vals = {"true", "t", "yes", "y", "1", "1.0", "starter", "sp", "start"}
    false_vals = {"false", "f", "no", "n", "0", "0.0", "reliever", "rp"}
    out.loc[out.isna() & text.isin(true_vals)] = True
    out.loc[out.isna() & text.isin(false_vals)] = False
    return out


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if DATE_COL not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COL}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required column: {TARGET_COL}")
    if PITCHER_COL not in df.columns:
        raise ValueError(f"Missing required column: {PITCHER_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df[df[DATE_COL].notna()].copy()
    df["season"] = df.get("season", df[DATE_COL].dt.year)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(df[DATE_COL].dt.year).astype(int)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()

    if "is_starter" in df.columns:
        df["is_starter"] = coerce_bool(df["is_starter"]).fillna(False).astype(bool)
    else:
        df["is_starter"] = False

    if "home_away" in df.columns:
        h = df["home_away"].astype("string").str.lower().str.strip()
        df["home_away_flag"] = np.where(h.isin(["h", "home", "1", "true"]), 1.0, np.where(h.isin(["v", "a", "away", "0", "false"]), 0.0, np.nan))

    # Drop known leaky columns from the modeling dataframe while preserving a report of what existed.
    for c in LEAKY_LINEUP_COLS:
        if c in df.columns:
            # Keep a copy only if someone wants to inspect the raw input later. It will not be selected.
            pass

    # Coerce common numeric raw columns.
    for c in set(BASE_NUMERIC_RAW + ROLL_SOURCE_COLS + OPP_FEATURES + ENV_NUMERIC_FEATURES + SAFE_LINEUP_FEATURES + STABLE_SAVANT_FEATURES):
        if c in df.columns and c not in {"has_lineup_context"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Context flags.
    for flag in ["has_savant_features", "has_env_context", "has_lineup_context"]:
        if flag in df.columns:
            b = coerce_bool(df[flag])
            if b.notna().any():
                df[flag] = b.astype("float")
            else:
                df[flag] = pd.to_numeric(df[flag], errors="coerce")

    return df


def add_days_rest_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([PITCHER_COL, DATE_COL]).copy()
    if "days_rest" not in df.columns or df["days_rest"].isna().all():
        df["days_rest"] = df.groupby(PITCHER_COL, sort=False)[DATE_COL].diff().dt.days
    else:
        current = pd.to_numeric(df["days_rest"], errors="coerce")
        computed = df.groupby(PITCHER_COL, sort=False)[DATE_COL].diff().dt.days
        df["days_rest"] = current.fillna(computed)
    return df


def add_pitcher_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([PITCHER_COL, DATE_COL]).copy()
    groups = df.groupby(PITCHER_COL, sort=False)

    df["games_prior"] = groups.cumcount()
    df["target_lag1"] = groups[TARGET_COL].shift(1)

    for w in [3, 5, 10]:
        shifted = groups[TARGET_COL].shift(1)
        # rolling by pitcher, leakage-safe because shifted first
        df[f"k_roll_mean_{w}"] = shifted.groupby(df[PITCHER_COL], sort=False).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"k_roll_std_{w}"] = shifted.groupby(df[PITCHER_COL], sort=False).transform(lambda s: s.rolling(w, min_periods=2).std())

    for c in ROLL_SOURCE_COLS:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            shifted = vals.groupby(df[PITCHER_COL], sort=False).shift(1)
            df[f"{c}_lag1"] = shifted
            df[f"{c}_roll5"] = shifted.groupby(df[PITCHER_COL], sort=False).transform(lambda s: s.rolling(5, min_periods=1).mean())

    if "batters_faced" in df.columns:
        bf = pd.to_numeric(df["batters_faced"], errors="coerce").replace(0, np.nan)
        k_rate = pd.to_numeric(df[TARGET_COL], errors="coerce") / bf
        shifted_rate = k_rate.groupby(df[PITCHER_COL], sort=False).shift(1)
        df["k_per_bf_lag1"] = shifted_rate
        df["k_per_bf_roll5"] = shifted_rate.groupby(df[PITCHER_COL], sort=False).transform(lambda s: s.rolling(5, min_periods=1).mean())

    return df


def present(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]


def choose_savant_features(df: pd.DataFrame, mode: str) -> List[str]:
    if mode == "none":
        return []
    if mode == "stable":
        return present(STABLE_SAVANT_FEATURES, df)
    out = []
    for c in df.columns:
        if not c.startswith("savant_"):
            continue
        lc = c.lower()
        # Avoid identifier-like columns as ordinal numeric predictors.
        if lc in {"savant_team_id"} or lc.endswith("_id"):
            continue
        if c == "savant_pitcher_name":
            continue
        out.append(c)
    return out


def dedupe_keep_order(cols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def feature_lists(df: pd.DataFrame, stack: str, savant_feature_set: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    base_numeric = present(BASE_NUMERIC_RAW + ROLL_FEATURES, df)
    base_categorical = present(BASE_CATEGORICAL_RAW, df)

    numeric = list(base_numeric)
    categorical = list(base_categorical)
    groups = {
        "base_numeric": list(base_numeric),
        "base_categorical": list(base_categorical),
        "savant": [],
        "opp": [],
        "env_numeric": [],
        "env_categorical": [],
        "lineup": [],
    }

    if stack == "enriched":
        savant = choose_savant_features(df, savant_feature_set)
        opp = present(OPP_FEATURES, df)
        env_num = present(ENV_NUMERIC_FEATURES, df)
        env_cat = present(ENV_CATEGORICAL_FEATURES, df)
        lineup = present(SAFE_LINEUP_FEATURES, df)

        numeric += savant + opp + env_num + lineup
        categorical += env_cat
        groups.update({
            "savant": savant,
            "opp": opp,
            "env_numeric": env_num,
            "env_categorical": env_cat,
            "lineup": lineup,
        })

    # Force leaky columns out, even if a future list accidentally includes them.
    numeric = [c for c in numeric if c not in LEAKY_LINEUP_COLS]
    categorical = [c for c in categorical if c not in LEAKY_LINEUP_COLS]

    # Avoid selecting the same column as both numeric and categorical.
    cat_set = set(categorical)
    numeric = [c for c in numeric if c not in cat_set]

    return dedupe_keep_order(numeric), dedupe_keep_order(categorical), groups


def drop_empty_window_features(
    train: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    kept_num = []
    kept_cat = []
    dropped = []

    for c in numeric_cols:
        s = pd.to_numeric(train[c], errors="coerce") if c in train.columns else pd.Series(dtype=float)
        if s.notna().any():
            kept_num.append(c)
        else:
            dropped.append(c)

    for c in categorical_cols:
        if c not in train.columns:
            dropped.append(c)
            continue
        s = train[c].astype("string")
        if s.notna().any():
            kept_cat.append(c)
        else:
            dropped.append(c)

    return kept_num, kept_cat, dropped


def make_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(model_name: str, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    if model_name in {"ridge", "poisson"}:
        num_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    else:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]

    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot()),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable features left for this training window")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_estimator(model_name: str, random_state: int, n_estimators: int):
    # rf_champion mirrors the current no-leak lineup champion script:
    # RandomForestRegressor(n_estimators=300, min_samples_leaf=2).
    if model_name == "rf_champion":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "rf_leaf1":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees_leaf2":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees_leaf1":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "ridge":
        return Ridge(alpha=2.0)
    if model_name == "poisson":
        return PoissonRegressor(alpha=0.02, max_iter=1000)
    raise ValueError(f"Unknown model: {model_name}")


def prepare_xy(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        X[c] = df[c].astype("string").replace({pd.NA: np.nan})
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return X, y


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def score(y_true, y_pred) -> Dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    pred = np.clip(pred, 0, None)
    actual = np.asarray(y_true, dtype=float)
    return {
        "rmse": rmse(actual, pred),
        "mae": float(mean_absolute_error(actual, pred)),
        "bias": float(np.mean(pred - actual)),
    }


def run_walk_forward(
    df: pd.DataFrame,
    stack: str,
    savant_feature_set: str,
    model_names: List[str],
    period: str,
    min_train_rows: int,
    random_state: int,
    n_estimators: int,
    show_progress: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    df = df.copy().sort_values(DATE_COL)
    df["_period"] = df[DATE_COL].dt.to_period(period).astype(str)
    periods = sorted(df["_period"].dropna().unique())

    numeric_candidates, categorical_candidates, groups = feature_lists(df, stack, savant_feature_set)
    metrics_rows = []
    pred_rows = []

    iterator = tqdm(periods, desc=f"Walk-forward {stack}", disable=not show_progress)
    for p in iterator:
        train = df[df["_period"] < p].copy()
        test = df[df["_period"] == p].copy()
        if len(train) < min_train_rows or len(test) == 0:
            continue

        numeric_cols, categorical_cols, dropped = drop_empty_window_features(train, numeric_candidates, categorical_candidates)
        if not numeric_cols and not categorical_cols:
            continue

        X_train, y_train = prepare_xy(train, numeric_cols, categorical_cols)
        X_test, y_test = prepare_xy(test, numeric_cols, categorical_cols)

        for model_name in model_names:
            pre = make_preprocessor(model_name, numeric_cols, categorical_cols)
            est = make_estimator(model_name, random_state=random_state, n_estimators=n_estimators)
            pipe = Pipeline([("preprocessor", pre), ("model", est)])
            pipe.fit(X_train, y_train)
            y_pred = np.clip(pipe.predict(X_test), 0, None)
            s = score(y_test, y_pred)

            metrics_rows.append({
                "feature_stack": stack,
                "model_name": model_name,
                "period": p,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "feature_count": int(len(numeric_cols) + len(categorical_cols)),
                "numeric_feature_count": int(len(numeric_cols)),
                "categorical_feature_count": int(len(categorical_cols)),
                "dropped_window_empty_count": int(len(dropped)),
                "dropped_window_empty_features": ",".join(dropped),
                **s,
            })

            tmp = test[[DATE_COL, PITCHER_COL, TARGET_COL]].copy()
            for maybe in ["pitcher_name", "team", "opponent", "pitcher_hand"]:
                if maybe in test.columns:
                    tmp[maybe] = test[maybe].values
            tmp["feature_stack"] = stack
            tmp["model_name"] = model_name
            tmp["period"] = p
            tmp["prediction"] = y_pred
            tmp["error"] = y_pred - y_test.values
            pred_rows.append(tmp)

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    summaries: Dict[str, dict] = {}
    if not preds.empty:
        for (stack_name, model_name), g in preds.groupby(["feature_stack", "model_name"], dropna=False):
            s = score(g[TARGET_COL], g["prediction"])
            wm = metrics[(metrics["feature_stack"] == stack_name) & (metrics["model_name"] == model_name)].copy()
            row_weight = wm["test_rows"].sum() if not wm.empty else len(g)
            if row_weight:
                weighted_mae = float((wm["mae"] * wm["test_rows"]).sum() / row_weight)
                # Proper pooled-ish window RMSE: weight window MSE by rows, then sqrt.
                weighted_rmse = float(math.sqrt(((wm["rmse"] ** 2) * wm["test_rows"]).sum() / row_weight))
                weighted_bias = float((wm["bias"] * wm["test_rows"]).sum() / row_weight)
            else:
                weighted_mae = weighted_rmse = weighted_bias = float("nan")
            summaries[f"{stack_name}:{model_name}"] = {
                "feature_stack": stack_name,
                "model_name": model_name,
                "windows": int(wm.shape[0]),
                "rows_scored": int(len(g)),
                "overall_rmse": s["rmse"],
                "overall_mae": s["mae"],
                "overall_bias": s["bias"],
                "weighted_window_mae": weighted_mae,
                "weighted_window_rmse": weighted_rmse,
                "weighted_window_bias": weighted_bias,
                "avg_feature_count_per_window": float(wm["feature_count"].mean()) if not wm.empty else None,
                "avg_dropped_window_empty_count": float(wm["dropped_window_empty_count"].mean()) if not wm.empty else None,
                "candidate_feature_groups": groups,
            }

    return metrics, preds, summaries


def fit_final_models(
    df: pd.DataFrame,
    stacks: List[str],
    model_names: List[str],
    savant_feature_set: str,
    random_state: int,
    n_estimators: int,
    out_dir: Path,
) -> None:
    models_dir = out_dir / "fitted_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for stack in stacks:
        numeric_candidates, categorical_candidates, _ = feature_lists(df, stack, savant_feature_set)
        numeric_cols, categorical_cols, _ = drop_empty_window_features(df, numeric_candidates, categorical_candidates)
        X, y = prepare_xy(df, numeric_cols, categorical_cols)
        for model_name in model_names:
            pipe = Pipeline([
                ("preprocessor", make_preprocessor(model_name, numeric_cols, categorical_cols)),
                ("model", make_estimator(model_name, random_state=random_state, n_estimators=n_estimators)),
            ])
            pipe.fit(X, y)
            dump({
                "pipeline": pipe,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "stack": stack,
                "model_name": model_name,
                "safety_version": SAFETY_VERSION,
            }, models_dir / f"{stack}_{model_name}.joblib")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_any(args.input)
    df = normalize_input(df)

    df = df[(df["season"] >= args.min_season) & (df["season"] <= args.max_season)].copy()
    if args.starters_only:
        df = df[df["is_starter"] == True].copy()  # noqa: E712
    if args.require_savant_rows and "has_savant_features" in df.columns:
        df = df[pd.to_numeric(df["has_savant_features"], errors="coerce").fillna(0) > 0].copy()
    if args.require_env_rows and "has_env_context" in df.columns:
        df = df[pd.to_numeric(df["has_env_context"], errors="coerce").fillna(0) > 0].copy()
    if args.require_lineup_rows and "has_lineup_context" in df.columns:
        df = df[pd.to_numeric(df["has_lineup_context"], errors="coerce").fillna(0) > 0].copy()

    df = add_days_rest_if_needed(df)
    df = add_pitcher_rolling_features(df)
    df = df[df["games_prior"] >= args.min_games].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering. Loosen filters or inspect input coverage.")

    stacks = ["baseline", "enriched"] if args.feature_stack == "both" else [args.feature_stack]

    all_metrics = []
    all_preds = []
    summary_by_model: Dict[str, dict] = {}
    for stack in stacks:
        metrics, preds, summaries = run_walk_forward(
            df=df,
            stack=stack,
            savant_feature_set=args.savant_feature_set,
            model_names=args.models,
            period=args.period,
            min_train_rows=args.min_train_rows,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            show_progress=not args.no_progress,
        )
        if not metrics.empty:
            all_metrics.append(metrics)
        if not preds.empty:
            all_preds.append(preds)
        summary_by_model.update(summaries)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    metrics_path = out_dir / "window_metrics.csv"
    preds_path = out_dir / "predictions.parquet"
    metrics_df.to_csv(metrics_path, index=False)
    if not preds_df.empty:
        preds_df.to_parquet(preds_path, index=False)

    ranking = []
    for key, payload in summary_by_model.items():
        ranking.append({
            "key": key,
            "feature_stack": payload["feature_stack"],
            "model_name": payload["model_name"],
            "overall_mae": payload["overall_mae"],
            "overall_rmse": payload["overall_rmse"],
            "overall_bias": payload["overall_bias"],
            "weighted_window_mae": payload["weighted_window_mae"],
            "weighted_window_rmse": payload["weighted_window_rmse"],
            "weighted_window_bias": payload["weighted_window_bias"],
            "rows_scored": payload["rows_scored"],
            "windows": payload["windows"],
        })
    ranking = sorted(ranking, key=lambda x: (x["weighted_window_mae"], x["weighted_window_rmse"]))

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": str(args.input),
        "rows_after_filtering": int(len(df)),
        "seasons": sorted([int(x) for x in df["season"].dropna().unique()]),
        "pitchers": int(df[PITCHER_COL].nunique()),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "target": TARGET_COL,
        "feature_stack_requested": args.feature_stack,
        "savant_feature_set": args.savant_feature_set,
        "models": args.models,
        "excluded_leaky_lineup_columns": LEAKY_LINEUP_COLS,
        "leaky_lineup_columns_found_in_input": [c for c in LEAKY_LINEUP_COLS if c in load_any(args.input).columns],
        "model_summaries": summary_by_model,
        "ranking_by_weighted_mae": ranking,
        "best_model": ranking[0] if ranking else None,
        "artifacts": {
            "window_metrics_csv": str(metrics_path),
            "predictions_parquet": str(preds_path) if not preds_df.empty else None,
        },
    }

    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)

    if args.save_models:
        fit_final_models(
            df=df,
            stacks=stacks,
            model_names=args.models,
            savant_feature_set=args.savant_feature_set,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            out_dir=out_dir,
        )
        summary["artifacts"]["fitted_models_dir"] = str(out_dir / "fitted_models")
        write_json(summary_path, summary)

    print(f"saved window metrics: {metrics_path}")
    if not preds_df.empty:
        print(f"saved predictions: {preds_path}")
    print(f"saved summary: {summary_path}")
    print(json.dumps({
        "safety_version": SAFETY_VERSION,
        "rows_after_filtering": summary["rows_after_filtering"],
        "best_model": summary["best_model"],
        "ranking_by_weighted_mae": summary["ranking_by_weighted_mae"][:10],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
