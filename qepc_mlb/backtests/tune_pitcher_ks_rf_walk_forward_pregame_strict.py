from __future__ import annotations

"""
QEPC-MLB pitcher strikeout PRE-GAME STRICT Random Forest walk-forward tuner.

This script tunes RandomForestRegressor hyperparameters under the same leakage-safe
walk-forward setup used by the current pitcher-K champion stack:
  - stable/all Savant pitcher-season features
  - opponent K context
  - environment context
  - safe lineup context only

It deliberately excludes same-game outcome/state columns such as:
  - lineup_batting_ks / lineup_batting_pa
  - innings_pitched / outs_recorded / batters_faced
  - walks / hits_allowed / earned_runs / runs_allowed

Important: by default, Savant features are disabled because same-season
leaderboard exports can include future information inside a walk-forward backtest.
Use --savant_mode prev_season to use prior-season Savant features only.

Typical run:
python qepc_mlb/backtests/tune_pitcher_ks_rf_walk_forward.py \
  --input cache/mlb/processed/pitcher_game_logs_enriched_opp_env_lineup.parquet \
  --feature_stack enriched \
  --min_season 2024 \
  --max_season 2025 \
  --starters_only \
  --min_games 5 \
  --require_savant_rows \
  --require_lineup_rows \
  --savant_feature_set stable
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TARGET_COL = "strikeouts"
SAFETY_VERSION = "rf_tune_pitcher_ks_pregame_strict_v2_savant_mode"

LEAKY_LINEUP_COLS = {
    "lineup_batting_ks",
    "lineup_batting_pa",
}

SAFE_LINEUP_CANDIDATES = [
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

CURRENT_GAME_LEAKY_FEATURES = {
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
}

# Pregame-safe base features only. Current-game pitcher outcomes/state are
# intentionally excluded and can only enter as shift(1) lag/rolling features.
BASE_NUMERIC_CANDIDATES = [
    "days_rest",
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

BASE_CATEGORICAL_CANDIDATES = [
    "pitcher_hand",
    "team",
    "opponent",
    "home_away",
]


@dataclass(frozen=True)
class RFConfig:
    name: str
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    min_samples_split: int = 2
    max_features: float | str | None = 1.0
    bootstrap: bool = True

    def params(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("name")
        return d


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")


def coerce_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    out.loc[out.isna() & text.isin(["true", "t", "1", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & text.isin(["false", "f", "0", "no", "n", "rp", "reliever"])] = False
    return out


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_date" not in df.columns:
        raise ValueError("Missing game_date")
    if "pitcher_id" not in df.columns:
        raise ValueError("Missing pitcher_id")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year
    else:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")

    if "is_starter" in df.columns:
        df["is_starter"] = coerce_bool(df["is_starter"])
    else:
        df["is_starter"] = pd.Series(False, index=df.index, dtype="boolean")

    boolish = [
        "has_savant_features",
        "has_env_context",
        "has_lineup_context",
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
    ]
    for c in boolish:
        if c in df.columns:
            df[c] = coerce_bool(df[c])

    numeric_like = set(BASE_NUMERIC_CANDIDATES + list(CURRENT_GAME_LEAKY_FEATURES) + [TARGET_COL])
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_pitcher_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe rolling pitcher features using shift(1)."""
    df = df.copy()
    df = df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    groups = df.groupby("pitcher_id", sort=False)

    df["games_prior"] = groups.cumcount()
    df["target_lag1"] = groups[TARGET_COL].shift(1)

    for window in [3, 5, 10]:
        df[f"k_roll_mean_{window}"] = groups[TARGET_COL].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"k_roll_std_{window}"] = groups[TARGET_COL].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    for base in [
        "batters_faced",
        "innings_pitched",
        "outs_recorded",
        "days_rest",
        "walks",
        "hits_allowed",
        "earned_runs",
        "runs_allowed",
    ]:
        if base in df.columns:
            df[f"{base}_lag1"] = groups[base].shift(1)
            df[f"{base}_roll5"] = groups[base].transform(
                lambda s: s.shift(1).rolling(5, min_periods=1).mean()
            )

    if "batters_faced" in df.columns:
        bf_lag = groups["batters_faced"].shift(1)
        k_lag = groups[TARGET_COL].shift(1)
        df["k_per_bf_lag1"] = k_lag / bf_lag.replace(0, np.nan)

        k_roll_sum = groups[TARGET_COL].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        bf_roll_sum = groups["batters_faced"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        df["k_per_bf_roll5"] = k_roll_sum / bf_roll_sum.replace(0, np.nan)

    return df


def stable_savant_features(df: pd.DataFrame) -> List[str]:
    candidates = [
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
    return [c for c in candidates if c in df.columns]


def all_savant_features(df: pd.DataFrame) -> List[str]:
    disallow = {"player_id", "pitcher_name_savant", "savant_team_id"}
    return [c for c in df.columns if c.startswith("savant_") and c not in disallow]


def opponent_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("opp_team_")]


def environment_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    categorical_exact = {"env_winddir", "env_daynight", "env_park_id", "env_usedh"}
    for c in df.columns:
        if not c.startswith("env_"):
            continue
        if c in categorical_exact:
            categorical.append(c)
        else:
            numeric.append(c)
    if "has_env_context" in df.columns:
        numeric.append("has_env_context")
    return numeric, categorical


def lineup_features(df: pd.DataFrame) -> List[str]:
    return [c for c in SAFE_LINEUP_CANDIDATES if c in df.columns and c not in LEAKY_LINEUP_COLS]



def normalize_name_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def first_non_null(series: pd.Series):
    vals = series.dropna()
    if vals.empty:
        return np.nan
    return vals.iloc[0]


def apply_savant_mode(df: pd.DataFrame, savant_mode: str, savant_feature_set: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Control temporal use of Savant features.

    none:       remove Savant features from candidate set.
    as_is:      use already-joined same-season Savant features. Useful only for diagnostics;
                it can leak future information inside a season.
    prev_season: replace Savant columns with previous-season values per pitcher. This is
                 the pregame-safe option when only full-season leaderboards are available.
    """
    df = df.copy()
    cols = stable_savant_features(df) if savant_feature_set == "stable" else all_savant_features(df)
    meta = {"savant_mode": savant_mode, "savant_columns_seen": cols, "savant_columns_used_temporally": []}
    if not cols or savant_mode in {"none", "as_is"}:
        if savant_mode == "none" and "has_savant_features" in df.columns:
            # Keep the existing flag for filtering if requested, but no savant cols will be selected.
            pass
        if savant_mode == "as_is":
            meta["warning"] = "as_is uses same-season Savant features and may leak future within-season information."
        return df, meta

    if savant_mode != "prev_season":
        raise ValueError(f"Unknown savant_mode: {savant_mode}")

    if "season" not in df.columns:
        raise ValueError("prev_season Savant mode requires a season column")

    if "pitcher_name" in df.columns:
        df["__savant_pitcher_key"] = normalize_name_series(df["pitcher_name"])
    else:
        df["__savant_pitcher_key"] = df["pitcher_id"].astype("string")

    season_table = (
        df[["__savant_pitcher_key", "season"] + cols]
        .dropna(subset=["__savant_pitcher_key", "season"])
        .groupby(["__savant_pitcher_key", "season"], as_index=False)
        .agg({c: first_non_null for c in cols})
    )
    season_table["season"] = pd.to_numeric(season_table["season"], errors="coerce") + 1

    # Remove same-season Savant values, then merge previous-season values back under
    # the original column names so downstream code stays simple.
    df = df.drop(columns=cols, errors="ignore")
    df = df.merge(season_table, how="left", on=["__savant_pitcher_key", "season"])
    df = df.drop(columns=["__savant_pitcher_key"], errors="ignore")

    if cols:
        df["has_prev_season_savant_features"] = df[cols].notna().any(axis=1)
        df["has_savant_features"] = df["has_prev_season_savant_features"]
    meta["savant_columns_used_temporally"] = cols
    return df, meta

def dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def prepare_feature_lists(
    df: pd.DataFrame,
    feature_stack: str,
    savant_feature_set: str,
    savant_mode: str,
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    numeric = [c for c in BASE_NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in BASE_CATEGORICAL_CANDIDATES if c in df.columns]

    meta: Dict[str, List[str]] = {
        "savant": [],
        "opponent": [],
        "environment_numeric": [],
        "environment_categorical": [],
        "lineup": [],
    }

    if feature_stack == "enriched":
        if savant_mode == "none":
            savant = []
        else:
            savant = stable_savant_features(df) if savant_feature_set == "stable" else all_savant_features(df)
        opp = opponent_features(df)
        env_num, env_cat = environment_features(df)
        lineup = lineup_features(df)

        numeric.extend(savant)
        numeric.extend(opp)
        numeric.extend(env_num)
        numeric.extend(lineup)
        categorical.extend(env_cat)

        meta = {
            "savant": savant,
            "opponent": opp,
            "environment_numeric": env_num,
            "environment_categorical": env_cat,
            "lineup": lineup,
        }

    numeric = dedupe(numeric)
    categorical = dedupe(categorical)
    numeric = [c for c in numeric if c not in set(categorical)]

    selected = set(numeric) | set(categorical)
    leaky_selected = sorted(selected & LEAKY_LINEUP_COLS)
    current_game_selected = sorted(selected & CURRENT_GAME_LEAKY_FEATURES)
    if leaky_selected:
        raise RuntimeError(f"Lineup leakage guard triggered. Leaky columns selected: {leaky_selected}")
    if current_game_selected:
        raise RuntimeError(f"Pregame leakage guard triggered. Current-game columns selected: {current_game_selected}")

    return numeric + categorical, numeric, categorical, meta


def split_window_features(
    train: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    dropped_empty = []
    kept_num = []
    for c in numeric_features:
        if c not in train.columns:
            dropped_empty.append(c)
            continue
        s = pd.to_numeric(train[c], errors="coerce")
        if s.notna().any():
            kept_num.append(c)
        else:
            dropped_empty.append(c)

    kept_cat = []
    for c in categorical_features:
        if c not in train.columns:
            dropped_empty.append(c)
            continue
        s = train[c].astype("string")
        if s.dropna().shape[0] > 0:
            kept_cat.append(c)
        else:
            dropped_empty.append(c)

    return kept_num, kept_cat, dropped_empty


def build_pipeline(config: RFConfig, numeric_features: List[str], categorical_features: List[str], random_state: int, n_jobs: int) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    if not transformers:
        raise ValueError("No usable features for this training window")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(
        **config.params(),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def month_periods(df: pd.DataFrame) -> List[pd.Period]:
    return sorted(df["game_date"].dt.to_period("M").dropna().unique().tolist())


def grid_configs(grid: str) -> List[RFConfig]:
    fast = [
        RFConfig("rf_leaf2_300", n_estimators=300, min_samples_leaf=2, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf1_300", n_estimators=300, min_samples_leaf=1, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf3_300", n_estimators=300, min_samples_leaf=3, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf2_sqrt_300", n_estimators=300, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        RFConfig("rf_leaf2_500", n_estimators=500, min_samples_leaf=2, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf2_depth12_300", n_estimators=300, min_samples_leaf=2, max_depth=12, max_features=1.0, bootstrap=True),
    ]
    balanced = fast + [
        RFConfig("rf_leaf4_400", n_estimators=400, min_samples_leaf=4, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_400", n_estimators=400, min_samples_leaf=5, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf1_sqrt_400", n_estimators=400, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        RFConfig("rf_leaf3_sqrt_400", n_estimators=400, min_samples_leaf=3, max_features="sqrt", bootstrap=True),
        RFConfig("rf_leaf2_depth16_400", n_estimators=400, min_samples_leaf=2, max_depth=16, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf3_depth16_400", n_estimators=400, min_samples_leaf=3, max_depth=16, max_features=1.0, bootstrap=True),
    ]
    wide = balanced + [
        RFConfig("rf_leaf1_600", n_estimators=600, min_samples_leaf=1, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf2_800", n_estimators=800, min_samples_leaf=2, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf2_log2_500", n_estimators=500, min_samples_leaf=2, max_features="log2", bootstrap=True),
        RFConfig("rf_leaf3_log2_500", n_estimators=500, min_samples_leaf=3, max_features="log2", bootstrap=True),
        RFConfig("rf_leaf2_noboot_500", n_estimators=500, min_samples_leaf=2, max_features=1.0, bootstrap=False),
        RFConfig("rf_leaf3_noboot_500", n_estimators=500, min_samples_leaf=3, max_features=1.0, bootstrap=False),
    ]

    # Fine grid: focused around the balanced-grid winner neighborhood.
    # The prior best was leaf=5, 400 trees, max_depth=None, max_features=1.0.
    fine = [
        RFConfig("rf_leaf4_400", n_estimators=400, min_samples_leaf=4, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_400", n_estimators=400, min_samples_leaf=5, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf6_400", n_estimators=400, min_samples_leaf=6, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf7_400", n_estimators=400, min_samples_leaf=7, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_400", n_estimators=400, min_samples_leaf=8, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf4_600", n_estimators=600, min_samples_leaf=4, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_600", n_estimators=600, min_samples_leaf=5, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf6_600", n_estimators=600, min_samples_leaf=6, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf7_600", n_estimators=600, min_samples_leaf=7, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_600", n_estimators=600, min_samples_leaf=8, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf4_depth14_500", n_estimators=500, min_samples_leaf=4, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_depth14_500", n_estimators=500, min_samples_leaf=5, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf6_depth14_500", n_estimators=500, min_samples_leaf=6, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf4_depth18_500", n_estimators=500, min_samples_leaf=4, max_depth=18, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_depth18_500", n_estimators=500, min_samples_leaf=5, max_depth=18, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf6_depth18_500", n_estimators=500, min_samples_leaf=6, max_depth=18, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf5_depth22_500", n_estimators=500, min_samples_leaf=5, max_depth=22, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf6_depth22_500", n_estimators=500, min_samples_leaf=6, max_depth=22, max_features=1.0, bootstrap=True),
    ]


    # Edge grid: the fine grid winner landed at the upper leaf boundary (leaf=8),
    # so this probes whether a slightly smoother forest generalizes better.
    edge = [
        RFConfig("rf_leaf8_300", n_estimators=300, min_samples_leaf=8, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_400", n_estimators=400, min_samples_leaf=8, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_600", n_estimators=600, min_samples_leaf=8, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf9_300", n_estimators=300, min_samples_leaf=9, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf9_400", n_estimators=400, min_samples_leaf=9, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf9_600", n_estimators=600, min_samples_leaf=9, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf10_300", n_estimators=300, min_samples_leaf=10, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf10_400", n_estimators=400, min_samples_leaf=10, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf10_600", n_estimators=600, min_samples_leaf=10, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf12_300", n_estimators=300, min_samples_leaf=12, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf12_400", n_estimators=400, min_samples_leaf=12, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf12_600", n_estimators=600, min_samples_leaf=12, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf14_400", n_estimators=400, min_samples_leaf=14, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf16_400", n_estimators=400, min_samples_leaf=16, max_depth=None, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_depth14_500", n_estimators=500, min_samples_leaf=8, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf10_depth14_500", n_estimators=500, min_samples_leaf=10, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf12_depth14_500", n_estimators=500, min_samples_leaf=12, max_depth=14, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf8_depth18_500", n_estimators=500, min_samples_leaf=8, max_depth=18, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf10_depth18_500", n_estimators=500, min_samples_leaf=10, max_depth=18, max_features=1.0, bootstrap=True),
        RFConfig("rf_leaf12_depth18_500", n_estimators=500, min_samples_leaf=12, max_depth=18, max_features=1.0, bootstrap=True),
    ]
    if grid == "fast":
        return fast
    if grid == "balanced":
        return balanced
    if grid == "fine":
        return fine
    if grid == "wide":
        return wide
    if grid == "edge":
        return edge
    raise ValueError(f"Unknown grid: {grid}")

def parse_only_configs(configs: List[RFConfig], only: str | None) -> List[RFConfig]:
    if not only:
        return configs
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    selected = [c for c in configs if c.name in wanted]
    missing = sorted(wanted - {c.name for c in selected})
    if missing:
        raise ValueError(f"Unknown config(s): {missing}. Available: {[c.name for c in configs]}")
    return selected


def fit_predict_window(
    config: RFConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    n_jobs: int,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    kept_num, kept_cat, dropped_empty = split_window_features(train, numeric_features, categorical_features)
    if not kept_num and not kept_cat:
        raise ValueError("No usable features after dropping all-empty columns")

    X_train = train[kept_num + kept_cat].copy()
    X_test = test[kept_num + kept_cat].copy()

    for c in kept_num:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    for c in kept_cat:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")

    y_train = pd.to_numeric(train[TARGET_COL], errors="coerce")
    pipe = build_pipeline(config, kept_num, kept_cat, random_state=random_state, n_jobs=n_jobs)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return preds, kept_num, kept_cat, dropped_empty


def run_config_walk_forward(
    df: pd.DataFrame,
    config: RFConfig,
    feature_stack: str,
    savant_feature_set: str,
    savant_mode: str,
    min_train_rows: int,
    random_state: int,
    n_jobs: int,
    show_progress: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    periods = month_periods(df)
    windows = periods[1:]
    iterator = windows
    if tqdm is not None and show_progress:
        iterator = tqdm(windows, desc=f"Tuning {config.name}")

    _, numeric_candidates, categorical_candidates, feature_meta = prepare_feature_lists(df, feature_stack, savant_feature_set, savant_mode)

    window_rows = []
    pred_parts = []

    for period in iterator:
        train = df[df["game_date"] < period.start_time].copy()
        test = df[df["game_date"].dt.to_period("M") == period].copy()
        if len(train) < min_train_rows or test.empty:
            continue

        try:
            preds, kept_num, kept_cat, dropped_empty = fit_predict_window(
                config=config,
                train=train,
                test=test,
                numeric_features=numeric_candidates,
                categorical_features=categorical_candidates,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except ValueError:
            continue

        y_test = pd.to_numeric(test[TARGET_COL], errors="coerce")
        metrics = metric_dict(y_test, preds)
        window_rows.append({
            "config": config.name,
            "window": str(period),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": int(len(kept_num) + len(kept_cat)),
            "numeric_feature_count": int(len(kept_num)),
            "categorical_feature_count": int(len(kept_cat)),
            "dropped_window_empty_count": int(len(dropped_empty)),
            "dropped_window_empty_features": ",".join(sorted(dropped_empty)),
            **metrics,
        })

        keep_cols = [c for c in ["game_date", "pitcher_id", "pitcher_name", "team", "opponent"] if c in test.columns]
        part = test[keep_cols].copy()
        part["config"] = config.name
        part["window"] = str(period)
        part["actual"] = y_test.values
        part["pred"] = preds
        pred_parts.append(part)

    wm = pd.DataFrame(window_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    if preds.empty or wm.empty:
        summary = {
            "config": config.name,
            "rows_scored": 0,
            "windows": 0,
            "overall_rmse": np.nan,
            "overall_mae": np.nan,
            "overall_bias": np.nan,
            "weighted_window_mae": np.nan,
            "weighted_window_rmse": np.nan,
            "weighted_window_bias": np.nan,
        }
    else:
        overall = metric_dict(preds["actual"], preds["pred"])
        weights = wm["test_rows"].astype(float)
        summary = {
            "config": config.name,
            "params": config.params(),
            "rows_scored": int(len(preds)),
            "windows": int(wm.shape[0]),
            "overall_rmse": overall["rmse"],
            "overall_mae": overall["mae"],
            "overall_bias": overall["bias"],
            "weighted_window_mae": float(np.average(wm["mae"], weights=weights)),
            "weighted_window_rmse": float(np.average(wm["rmse"], weights=weights)),
            "weighted_window_bias": float(np.average(wm["bias"], weights=weights)),
            "avg_feature_count_per_window": float(wm["feature_count"].mean()),
            "avg_dropped_window_empty_count": float(wm["dropped_window_empty_count"].mean()),
        }

    summary["feature_meta"] = feature_meta
    return wm, preds, summary


def filter_df(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()

    leaked_present = sorted([c for c in LEAKY_LINEUP_COLS if c in df.columns])
    if leaked_present:
        df = df.drop(columns=leaked_present)

    if args.min_season is not None:
        df = df[df["season"] >= args.min_season].copy()
    if args.max_season is not None:
        df = df[df["season"] <= args.max_season].copy()
    if args.starters_only:
        df = df[df["is_starter"].fillna(False)].copy()
    if args.require_savant_rows and "has_savant_features" in df.columns:
        df = df[df["has_savant_features"].fillna(False)].copy()
    if args.require_lineup_rows and "has_lineup_context" in df.columns:
        df = df[df["has_lineup_context"].fillna(False)].copy()
    if args.require_env_rows and "has_env_context" in df.columns:
        df = df[df["has_env_context"].fillna(False)].copy()

    df = df[df["games_prior"] >= args.min_games].copy()
    df = df.dropna(subset=["game_date", TARGET_COL, "pitcher_id"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune Random Forest pitcher-K model with leakage-safe walk-forward evaluation.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="artifacts/mlb/backtests/pitcher_strikeouts_rf_tuning")
    ap.add_argument("--feature_stack", choices=["baseline", "enriched"], default="enriched")
    ap.add_argument("--savant_feature_set", choices=["stable", "all"], default="stable")
    ap.add_argument("--savant_mode", choices=["none", "as_is", "prev_season"], default="none", help="Temporal Savant handling: none=do not use Savant features; as_is=use same-season joined Savant features (diagnostic/leaky); prev_season=use prior-season Savant features only.")
    ap.add_argument("--grid", choices=["fast", "balanced", "fine", "wide", "edge"], default="fast")
    ap.add_argument("--only_configs", default=None, help="Optional comma-separated subset of config names from the selected grid.")
    ap.add_argument("--min_season", type=int, default=None)
    ap.add_argument("--max_season", type=int, default=None)
    ap.add_argument("--starters_only", action="store_true")
    ap.add_argument("--min_games", type=int, default=5)
    ap.add_argument("--require_savant_rows", action="store_true")
    ap.add_argument("--require_lineup_rows", action="store_true")
    ap.add_argument("--require_env_rows", action="store_true")
    ap.add_argument("--min_train_rows", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--save_predictions", action="store_true")
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_any(args.input)
    df = ensure_core(df_raw)
    df = compute_pitcher_rolls(df)
    df, savant_mode_meta = apply_savant_mode(df, args.savant_mode, args.savant_feature_set)

    leaked_present_before_filter = sorted([c for c in LEAKY_LINEUP_COLS if c in df.columns])
    df = filter_df(df, args)

    if df.empty:
        raise ValueError("No rows after filtering. Relax filters or inspect the input table.")

    configs = parse_only_configs(grid_configs(args.grid), args.only_configs)

    all_window_metrics = []
    all_predictions = []
    summaries = []

    cfg_iterator = configs
    if tqdm is not None and not args.no_progress:
        cfg_iterator = tqdm(configs, desc="RF configs")

    for cfg in cfg_iterator:
        wm, preds, summ = run_config_walk_forward(
            df=df,
            config=cfg,
            feature_stack=args.feature_stack,
            savant_feature_set=args.savant_feature_set,
            savant_mode=args.savant_mode,
            min_train_rows=args.min_train_rows,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            show_progress=not args.no_progress,
        )
        all_window_metrics.append(wm)
        if args.save_predictions and not preds.empty:
            all_predictions.append(preds)
        summaries.append(summ)

    window_metrics = pd.concat(all_window_metrics, ignore_index=True) if all_window_metrics else pd.DataFrame()
    ranking = pd.DataFrame([
        {k: v for k, v in s.items() if k not in {"feature_meta"}}
        for s in summaries
    ])
    if not ranking.empty:
        ranking = ranking.sort_values(["weighted_window_mae", "weighted_window_rmse"], ascending=True).reset_index(drop=True)

    best = ranking.iloc[0].to_dict() if not ranking.empty else {}

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": str(args.input),
        "feature_stack": args.feature_stack,
        "savant_feature_set": args.savant_feature_set,
        "savant_mode": args.savant_mode,
        "savant_mode_meta": savant_mode_meta,
        "grid": args.grid,
        "rows_after_filtering": int(len(df)),
        "seasons": sorted(pd.Series(df["season"]).dropna().astype(int).unique().tolist()),
        "pitchers": int(df["pitcher_id"].nunique()),
        "leaky_lineup_columns_found_and_dropped_from_input": leaked_present_before_filter,
        "globally_excluded_leaky_lineup_features": sorted(LEAKY_LINEUP_COLS),
        "configs_tested": [c.name for c in configs],
        "best_config": best,
        "ranking_by_weighted_mae": ranking.to_dict(orient="records") if not ranking.empty else [],
        "feature_meta": summaries[0].get("feature_meta", {}) if summaries else {},
    }

    window_metrics_path = out_dir / "window_metrics.csv"
    ranking_path = out_dir / "ranking.csv"
    summary_path = out_dir / "summary.json"

    window_metrics.to_csv(window_metrics_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    if args.save_predictions:
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        if not predictions.empty:
            predictions.to_parquet(out_dir / "predictions.parquet", index=False)

    print(f"saved window metrics: {window_metrics_path}")
    print(f"saved ranking: {ranking_path}")
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
