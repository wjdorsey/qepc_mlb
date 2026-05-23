#!/usr/bin/env python3
"""
Tune QEPC-MLB batter 1+ hit probability/ranking models.

Primary goal:
    Find models that rank the most likely hitters to record 1+ hit.

Important safety design:
    - Excludes same-game outcome columns such as hits, at_bats, plate_appearances.
    - Uses rolling/lag/season-prior hitter features only.
    - Optional confirmed lineup features are allowed only when --confirmed_lineups is passed.
    - Walk-forward evaluation trains only on rows before each test period.

Example:
    python qepc_mlb/backtests/tune_batter_1plus_hit_ranker.py \
      --input cache/mlb/processed/batter_game_logs.parquet \
      --min_season 2022 \
      --max_season 2025 \
      --starters_only \
      --confirmed_lineups \
      --min_games 20 \
      --grid hgb_focus
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SAFETY_VERSION = "batter_1plus_hit_ranker_v2_hgb_focus"
TARGET = "hit_1plus"

# Same-game outcomes / diagnostics that must never be model features.
EXCLUDED_SAME_GAME_OUTCOME_FEATURES = sorted({
    "hit_1plus",
    "hits",
    "b_h",
    "at_bats",
    "b_ab",
    "plate_appearances",
    "b_pa",
    "runs",
    "b_r",
    "rbi",
    "b_rbi",
    "doubles",
    "b_d",
    "triples",
    "b_t",
    "home_runs",
    "b_hr",
    "walks",
    "b_w",
    "strikeouts",
    "b_k",
    "hit_by_pitch",
    "b_hbp",
    "stolen_bases",
    "b_sb",
    "caught_stealing",
    "b_cs",
    "grounded_into_double_play",
    "b_gdp",
    "sacrifice_hits",
    "b_sh",
    "sacrifice_flies",
    "b_sf",
    "total_bases",
})

STARTER_ALIASES = [
    "is_starter",
    "is_starting_lineup",
    "is_starting_lineup_teamstats",
    "is_starting_lineup_by_batting_file",
    "starter_flag",
    "confirmed_starter",
]

LINEUP_FEATURES = [
    "lineup_slot",
    "is_starting_lineup",
    "is_starting_lineup_by_batting_file",
    "is_starting_lineup_teamstats",
    "is_starter",
]

BASE_NUMERIC_FEATURES = [
    "is_home",
    "env_temp",
    "env_windspeed",
    "has_env_context",
    "batter_games_prior",
    "games_prior",
    "batter_days_rest",
]

ENV_CATEGORICAL_FEATURES = [
    "env_winddir",
    "env_daynight",
    "env_usedh",
]

BASE_CATEGORICAL_FEATURES = [
    "batter_hand",
    "team",
    "opponent",
    "home_away",
] + ENV_CATEGORICAL_FEATURES


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}")


def ensure_season_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_date" not in df.columns:
        for c in ["date", "game_dt"]:
            if c in df.columns:
                df["game_date"] = df[c]
                break
    if "game_date" not in df.columns:
        raise ValueError("Missing game_date/date column.")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def coerce_boolish(s: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(False, index=s.index, dtype=bool)
    out.loc[numeric.fillna(0) != 0] = True
    text = s.astype("string").str.strip().str.lower()
    true_vals = {"true", "t", "yes", "y", "starter", "start", "starting", "1", "1.0"}
    false_vals = {"false", "f", "no", "n", "0", "0.0", "nan", "<na>", "none", ""}
    out.loc[text.isin(true_vals)] = True
    out.loc[text.isin(false_vals)] = False
    return out


def ensure_starter_alias(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    found = None
    for c in STARTER_ALIASES:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(
            "No starter/lineup flag found. Looked for: " + ", ".join(STARTER_ALIASES)
        )
    df["is_starter"] = coerce_boolish(df[found])
    return df, found


def add_games_prior_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "batter_id" not in df.columns:
        for c in ["id", "player_id"]:
            if c in df.columns:
                df["batter_id"] = df[c]
                break
    if "batter_id" not in df.columns:
        raise ValueError("Missing batter_id/id column.")
    df = df.sort_values(["batter_id", "game_date", "game_id" if "game_id" in df.columns else "batter_id"]).copy()
    if "batter_games_prior" not in df.columns:
        df["batter_games_prior"] = df.groupby("batter_id").cumcount()
    if "games_prior" not in df.columns:
        df["games_prior"] = df["batter_games_prior"]
    return df


def filter_rows(
    df: pd.DataFrame,
    min_season: Optional[int],
    max_season: Optional[int],
    starters_only: bool,
    confirmed_lineups: bool,
    min_games: int,
    require_env_rows: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = ensure_season_and_dates(df)
    df, starter_source = ensure_starter_alias(df)
    df = add_games_prior_if_needed(df)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    mask = df["game_date"].notna() & df[TARGET].isin([0, 1])
    if min_season is not None:
        mask &= df["season"] >= min_season
    if max_season is not None:
        mask &= df["season"] <= max_season
    if starters_only:
        mask &= df["is_starter"].fillna(False).astype(bool)
    if confirmed_lineups:
        # At minimum, keep rows with a lineup slot or starter flag. In Retrosheet batter rows,
        # b_lp / lineup_slot is available for starters and substitutes.
        if "lineup_slot" in df.columns:
            mask &= pd.to_numeric(df["lineup_slot"], errors="coerce").notna()
        else:
            mask &= df["is_starter"].fillna(False).astype(bool)
    if min_games > 0:
        mask &= pd.to_numeric(df["batter_games_prior"], errors="coerce").fillna(0) >= min_games
    if require_env_rows and "has_env_context" in df.columns:
        mask &= coerce_boolish(df["has_env_context"])

    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError("No rows left after filtering. Relax filters or inspect schema.")

    meta = {
        "starter_source_column": starter_source,
        "rows_after_filtering": int(len(out)),
        "batters": int(out["batter_id"].nunique()) if "batter_id" in out.columns else None,
        "seasons": [int(x) for x in sorted(out["season"].dropna().unique())],
        "date_min": str(out["game_date"].min().date()),
        "date_max": str(out["game_date"].max().date()),
        "hit_1plus_rate": float(out[TARGET].mean()),
        "confirmed_lineups": bool(confirmed_lineups),
        "starters_only": bool(starters_only),
        "min_games": int(min_games),
    }
    return out, meta


def infer_features(df: pd.DataFrame, confirmed_lineups: bool) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    cols = set(df.columns)

    rolling_features = []
    for c in df.columns:
        lc = c.lower()
        if c in EXCLUDED_SAME_GAME_OUTCOME_FEATURES:
            continue
        if any(token in lc for token in ["_lag", "_roll", "season_prior", "games_prior", "days_rest"]):
            rolling_features.append(c)

    numeric = []
    for c in BASE_NUMERIC_FEATURES + rolling_features:
        if c in cols and c not in EXCLUDED_SAME_GAME_OUTCOME_FEATURES:
            numeric.append(c)

    if confirmed_lineups:
        for c in LINEUP_FEATURES:
            if c in cols and c not in EXCLUDED_SAME_GAME_OUTCOME_FEATURES:
                numeric.append(c)

    # Preserve order and dedupe.
    seen = set()
    numeric = [c for c in numeric if not (c in seen or seen.add(c))]

    # Only use columns that can be numeric-ish.
    valid_numeric = []
    for c in numeric:
        if c not in df.columns:
            continue
        # allow bools; otherwise require at least one numeric value
        if pd.api.types.is_bool_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any():
            valid_numeric.append(c)

    categorical = [c for c in BASE_CATEGORICAL_FEATURES if c in cols and c not in EXCLUDED_SAME_GAME_OUTCOME_FEATURES]
    # Avoid duplicate columns across transformers.
    categorical = [c for c in categorical if c not in valid_numeric]

    meta = {
        "hitter_recent_form": [c for c in valid_numeric if any(tok in c.lower() for tok in ["lag", "roll", "season_prior", "games_prior", "days_rest"])],
        "environment": [c for c in valid_numeric + categorical if c.startswith("env_") or c in {"is_home", "has_env_context"}],
        "confirmed_lineup": [c for c in valid_numeric if c in LINEUP_FEATURES],
        "categorical": categorical,
    }
    return valid_numeric, categorical, meta


def dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str], scale_numeric: bool = False) -> ColumnTransformer:
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", dense_one_hot_encoder()),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), list(numeric_features)),
            ("cat", Pipeline(cat_steps), list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


@dataclass
class ModelConfig:
    key: str
    model: Any
    scale_numeric: bool = False


def get_model_configs(grid: str, models: Optional[List[str]], random_state: int, n_jobs: int) -> List[ModelConfig]:
    configs: List[ModelConfig] = []

    def allow(name: str) -> bool:
        return not models or name in models

    if grid == "hgb_focus":
        # Focused grid around the first-pass winners:
        # - hgb_lr05_leaf31_l2 won top-10 daily hit rate
        # - hgb_lr05_leaf15_l2 won Brier/log-loss
        if allow("hgb"):
            for lr, iters in [(0.04, 320), (0.05, 280), (0.06, 240)]:
                for leaf in [15, 23, 31, 47]:
                    for l2 in [0.0, 0.005, 0.01, 0.025]:
                        configs.append(ModelConfig(
                            f"hgb_lr{str(lr).replace('.', '')}_leaf{leaf}_l2{str(l2).replace('.', '')}",
                            HistGradientBoostingClassifier(
                                max_iter=iters,
                                learning_rate=lr,
                                max_leaf_nodes=leaf,
                                l2_regularization=l2,
                                random_state=random_state,
                            ),
                        ))
        # Optional comparison configs if requested.
        if allow("rf"):
            configs += [
                ModelConfig("rf_500_leaf16", RandomForestClassifier(n_estimators=500, min_samples_leaf=16, max_depth=14, max_features=1.0, bootstrap=True, random_state=random_state, n_jobs=n_jobs)),
            ]
        if allow("extra_trees"):
            configs += [
                ModelConfig("extra_trees_400_leaf10", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, max_depth=None, max_features=1.0, random_state=random_state, n_jobs=n_jobs)),
            ]
    elif grid == "fast":
        if allow("hgb"):
            configs += [
                ModelConfig("hgb_lr05_leaf31_l2", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.0, random_state=random_state)),
                ModelConfig("hgb_lr03_leaf31_l2", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=0.01, random_state=random_state)),
                ModelConfig("hgb_lr05_leaf15_l2", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.01, random_state=random_state)),
            ]
        if allow("rf"):
            configs += [
                ModelConfig("rf_400_leaf10", RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_depth=None, max_features=1.0, bootstrap=True, random_state=random_state, n_jobs=n_jobs)),
                ModelConfig("rf_500_leaf16", RandomForestClassifier(n_estimators=500, min_samples_leaf=16, max_depth=14, max_features=1.0, bootstrap=True, random_state=random_state, n_jobs=n_jobs)),
            ]
        if allow("extra_trees"):
            configs += [
                ModelConfig("extra_trees_400_leaf10", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, max_depth=None, max_features=1.0, random_state=random_state, n_jobs=n_jobs)),
            ]
    else:
        if allow("hgb"):
            for lr, iters in [(0.02, 500), (0.03, 400), (0.05, 300), (0.07, 220)]:
                for leaf in [15, 31, 63]:
                    for l2 in [0.0, 0.01, 0.05, 0.1]:
                        configs.append(ModelConfig(
                            f"hgb_lr{str(lr).replace('.', '')}_leaf{leaf}_l2{str(l2).replace('.', '')}",
                            HistGradientBoostingClassifier(max_iter=iters, learning_rate=lr, max_leaf_nodes=leaf, l2_regularization=l2, random_state=random_state),
                        ))
        if allow("rf"):
            for leaf in [5, 10, 16, 25]:
                for trees in [400, 600]:
                    configs.append(ModelConfig(
                        f"rf_{trees}_leaf{leaf}",
                        RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf, max_depth=None, max_features=1.0, bootstrap=True, random_state=random_state, n_jobs=n_jobs),
                    ))
            for leaf, depth in [(10, 12), (10, 16), (16, 14), (25, 14)]:
                configs.append(ModelConfig(
                    f"rf_500_leaf{leaf}_depth{depth}",
                    RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_depth=depth, max_features=1.0, bootstrap=True, random_state=random_state, n_jobs=n_jobs),
                ))
        if allow("extra_trees"):
            for leaf in [5, 10, 16, 25]:
                configs.append(ModelConfig(
                    f"extra_trees_500_leaf{leaf}",
                    ExtraTreesClassifier(n_estimators=500, min_samples_leaf=leaf, max_depth=None, max_features=1.0, random_state=random_state, n_jobs=n_jobs),
                ))
        if allow("logreg"):
            for c in [0.1, 0.3, 1.0, 3.0]:
                configs.append(ModelConfig(
                    f"logreg_c{str(c).replace('.', '')}",
                    LogisticRegression(C=c, max_iter=2000, n_jobs=n_jobs, random_state=random_state),
                    scale_numeric=True,
                ))

    if not configs:
        raise ValueError("No model configs selected. Check --models values.")
    return configs


def prepare_xy(df: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = list(numeric_features) + list(categorical_features)
    X = df[feature_cols].copy()
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_features:
        X[c] = X[c].astype("string").fillna("__MISSING__")
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    return X, y


def safe_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred50 = (prob >= 0.5).astype(int)
    out: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "accuracy_50": float(accuracy_score(y_true, pred50)),
        "actual_rate": float(np.mean(y_true)),
        "avg_pred_prob": float(np.mean(prob)),
        "prob_bias": float(np.mean(prob) - np.mean(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["avg_precision"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["avg_precision"] = float("nan")
    return out


def topk_daily_metrics(pred_df: pd.DataFrame, ks: Sequence[int] = (5, 10, 25, 50)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if pred_df.empty:
        for k in ks:
            out[f"top{k}_daily_hit_rate"] = float("nan")
            out[f"top{k}_daily_dates"] = 0
            out[f"top{k}_daily_rows"] = 0
        return out
    temp = pred_df.copy()
    temp["game_date"] = pd.to_datetime(temp["game_date"], errors="coerce").dt.date
    for k in ks:
        top = temp.sort_values(["game_date", "pred_prob"], ascending=[True, False]).groupby("game_date", group_keys=False).head(k)
        out[f"top{k}_daily_hit_rate"] = float(top["actual"].mean()) if len(top) else float("nan")
        out[f"top{k}_daily_dates"] = int(top["game_date"].nunique()) if len(top) else 0
        out[f"top{k}_daily_rows"] = int(len(top))
    return out


def summarize_predictions(pred_df: pd.DataFrame, windows: int) -> Dict[str, Any]:
    if pred_df.empty:
        return {}
    m = safe_metrics(pred_df["actual"].to_numpy(), pred_df["pred_prob"].to_numpy())
    m.update(topk_daily_metrics(pred_df))
    m["rows_scored"] = int(len(pred_df))
    m["windows"] = int(windows)
    return m


def walk_forward_for_config(
    df: pd.DataFrame,
    config: ModelConfig,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    period: str,
    min_train_rows: int,
    show_progress: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_date").copy()
    df["period_key"] = df["game_date"].dt.to_period(period)
    periods = sorted(df["period_key"].dropna().unique())

    preds: List[pd.DataFrame] = []
    window_rows: List[Dict[str, Any]] = []

    iterator: Iterable[Any] = periods
    if show_progress and tqdm is not None:
        iterator = tqdm(periods, desc=f"Walk-forward {config.key}")

    for p in iterator:
        test_mask = df["period_key"] == p
        test = df.loc[test_mask].copy()
        if test.empty:
            continue
        period_start = test["game_date"].min()
        train = df.loc[df["game_date"] < period_start].copy()
        if len(train) < min_train_rows:
            continue
        if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 1:
            continue

        # Drop features that are entirely empty in this training window.
        use_num = [c for c in numeric_features if c in train.columns and pd.to_numeric(train[c], errors="coerce").notna().any()]
        use_cat = [c for c in categorical_features if c in train.columns and train[c].astype("string").notna().any()]
        if not use_num and not use_cat:
            continue

        X_train, y_train = prepare_xy(train, use_num, use_cat)
        X_test, y_test = prepare_xy(test, use_num, use_cat)

        pre = make_preprocessor(use_num, use_cat, scale_numeric=config.scale_numeric)
        pipe = Pipeline([("preprocess", pre), ("model", config.model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]

        pm = safe_metrics(y_test.to_numpy(), prob)
        pred = pd.DataFrame({
            "model": config.key,
            "game_date": test["game_date"].values,
            "season": test["season"].values,
            "batter_id": test.get("batter_id", pd.Series(index=test.index, dtype="object")).values,
            "batter_name": test.get("batter_name", pd.Series(index=test.index, dtype="object")).values,
            "team": test.get("team", pd.Series(index=test.index, dtype="object")).values,
            "opponent": test.get("opponent", pd.Series(index=test.index, dtype="object")).values,
            "lineup_slot": test.get("lineup_slot", pd.Series(index=test.index, dtype="float")).values,
            "actual": y_test.values,
            "pred_prob": prob,
            "period": str(p),
        })
        preds.append(pred)

        row = {
            "model": config.key,
            "period": str(p),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": int(len(use_num) + len(use_cat)),
            "numeric_feature_count": int(len(use_num)),
            "categorical_feature_count": int(len(use_cat)),
        }
        row.update(pm)
        row.update(topk_daily_metrics(pred, ks=(5, 10, 25)))
        window_rows.append(row)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    win_df = pd.DataFrame(window_rows)
    summary = summarize_predictions(pred_df, windows=len(win_df))
    summary["model"] = config.key
    summary["avg_feature_count_per_window"] = float(win_df["feature_count"].mean()) if not win_df.empty else float("nan")
    return summary, pred_df, win_df


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune batter 1+ hit ranker models with walk-forward evaluation.")
    p.add_argument("--input", required=True, help="Path to batter_game_logs parquet/csv.")
    p.add_argument("--out_dir", default="artifacts/mlb/backtests/batter_1plus_hit_ranker")
    p.add_argument("--min_season", type=int, default=None)
    p.add_argument("--max_season", type=int, default=None)
    p.add_argument("--starters_only", action="store_true")
    p.add_argument("--confirmed_lineups", action="store_true", help="Allow confirmed lineup features such as lineup_slot/starter flag.")
    p.add_argument("--min_games", type=int, default=20)
    p.add_argument("--require_env_rows", action="store_true")
    p.add_argument("--grid", choices=["fast", "wide", "hgb_focus"], default="fast")
    p.add_argument("--models", nargs="*", default=None, choices=["rf", "hgb", "extra_trees", "logreg"])
    p.add_argument("--period", default="M", help="Pandas period alias; M=monthly.")
    p.add_argument("--min_train_rows", type=int, default=5000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input)
    df, input_meta = filter_rows(
        df,
        min_season=args.min_season,
        max_season=args.max_season,
        starters_only=args.starters_only,
        confirmed_lineups=args.confirmed_lineups,
        min_games=args.min_games,
        require_env_rows=args.require_env_rows,
    )

    numeric_features, categorical_features, feature_meta = infer_features(df, confirmed_lineups=args.confirmed_lineups)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features found.")

    configs = get_model_configs(args.grid, args.models, args.random_state, args.n_jobs)

    summaries: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    win_frames: List[pd.DataFrame] = []

    for cfg in configs:
        summary, pred_df, win_df = walk_forward_for_config(
            df,
            cfg,
            numeric_features,
            categorical_features,
            period=args.period,
            min_train_rows=args.min_train_rows,
            show_progress=not args.no_progress,
        )
        if summary.get("rows_scored", 0):
            summaries.append(summary)
            if not pred_df.empty:
                pred_frames.append(pred_df)
            if not win_df.empty:
                win_frames.append(win_df)

    if not summaries:
        raise ValueError("No model produced scored rows. Try reducing --min_train_rows or checking date filters.")

    ranking_top10 = sorted(summaries, key=lambda x: (-x.get("top10_daily_hit_rate", -1), x.get("brier", math.inf)))
    ranking_brier = sorted(summaries, key=lambda x: (x.get("brier", math.inf), -x.get("top10_daily_hit_rate", -1)))
    ranking_log_loss = sorted(summaries, key=lambda x: (x.get("log_loss", math.inf), -x.get("top10_daily_hit_rate", -1)))

    all_windows = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame()
    all_preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    if not all_windows.empty:
        all_windows.to_csv(out_dir / "window_metrics.csv", index=False)
    if args.save_predictions and not all_preds.empty:
        all_preds.to_parquet(out_dir / "predictions.parquet", index=False)
        # CSV can be large but handy for inspection.
        all_preds.to_csv(out_dir / "predictions.csv", index=False)

    # Always save compact top-board predictions for the best ranker.
    best_key = ranking_top10[0]["model"]
    if not all_preds.empty:
        best_preds = all_preds[all_preds["model"] == best_key].copy()
        best_preds["game_date"] = pd.to_datetime(best_preds["game_date"], errors="coerce").dt.date
        top_board = best_preds.sort_values(["game_date", "pred_prob"], ascending=[True, False]).groupby("game_date", group_keys=False).head(25)
        top_board.to_csv(out_dir / "best_model_top25_daily_board.csv", index=False)

    final_summary = {
        "safety_version": SAFETY_VERSION,
        "input": args.input,
        "out_dir": str(out_dir),
        "target": TARGET,
        "objective_primary": "maximize top10_daily_hit_rate; use Brier/log_loss as calibration safeguards",
        "input_summary": input_meta,
        "grid": args.grid,
        "models_tested": [cfg.key for cfg in configs],
        "feature_counts": {
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
            "total": len(numeric_features) + len(categorical_features),
        },
        "feature_meta": feature_meta,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_same_game_outcome_features": EXCLUDED_SAME_GAME_OUTCOME_FEATURES,
        "best_model_by_top10_daily_hit_rate": ranking_top10[0],
        "best_model_by_weighted_brier": ranking_brier[0],
        "best_model_by_log_loss": ranking_log_loss[0],
        "ranking_by_top10_daily_hit_rate": ranking_top10,
        "ranking_by_weighted_brier": ranking_brier,
        "ranking_by_log_loss": ranking_log_loss,
        "outputs": {
            "window_metrics": str(out_dir / "window_metrics.csv"),
            "best_model_top25_daily_board": str(out_dir / "best_model_top25_daily_board.csv"),
            "predictions": str(out_dir / "predictions.parquet") if args.save_predictions else None,
            "summary": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", final_summary)

    print(json.dumps({
        "safety_version": SAFETY_VERSION,
        "best_model_by_top10_daily_hit_rate": ranking_top10[0],
        "best_model_by_weighted_brier": ranking_brier[0],
        "top5_ranking_by_top10": ranking_top10[:5],
        "outputs": final_summary["outputs"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
