from __future__ import annotations

"""
QEPC-MLB Pitcher Strikeouts Champion Trainer

Production-style trainer for the current no-leak pitcher-K champion stack.

Current champion config from walk-forward RF tuning:
  RandomForestRegressor(
      n_estimators=500,
      max_depth=14,
      min_samples_leaf=10,
      min_samples_split=2,
      max_features=1.0,
      bootstrap=True,
  )

Expected input:
  cache/mlb/processed/pitcher_game_logs_enriched_opp_env_lineup.parquet

Typical run:
python qepc_mlb/models/train_pitcher_ks_champion.py \
  --input cache/mlb/processed/pitcher_game_logs_enriched_opp_env_lineup.parquet \
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
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
SAFETY_VERSION = "pitcher_ks_champion_train_noleak_v1_rf_leaf10_depth14_500"

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

BASE_NUMERIC_CANDIDATES = [
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

BASE_CATEGORICAL_CANDIDATES = [
    "pitcher_hand",
    "team",
    "opponent",
]


@dataclass(frozen=True)
class ChampionRFConfig:
    n_estimators: int = 500
    max_depth: int | None = 14
    min_samples_leaf: int = 10
    min_samples_split: int = 2
    max_features: float | str | None = 1.0
    bootstrap: bool = True

    def params(self) -> Dict[str, Any]:
        return asdict(self)


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}")


def coerce_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    out.loc[out.isna() & text.isin(["true", "t", "1", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & text.isin(["false", "f", "0", "no", "n", "rp", "reliever"])] = False
    return out


def dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["game_date", "pitcher_id", TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Hard no-leak guard: physically remove same-game lineup outcome columns.
    for c in LEAKY_LINEUP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

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

    for c in set(BASE_NUMERIC_CANDIDATES + [TARGET_COL]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def compute_pitcher_rolls(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Create leakage-safe rolling pitcher features using shift(1)."""
    df = df.copy()
    df = df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    groups = df.groupby("pitcher_id", sort=False)

    df["games_prior"] = groups.cumcount()
    df["target_lag1"] = groups[TARGET_COL].shift(1)

    # These transform calls are fast enough for this table, but the stage print/progress
    # keeps the workflow consistent with the rest of QEPC-MLB.
    if tqdm is not None and show_progress:
        iterator = tqdm([3, 5, 10], desc="Rolling K windows")
    else:
        iterator = [3, 5, 10]

    for window in iterator:
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
    return dedupe(numeric), dedupe(categorical)


def lineup_features(df: pd.DataFrame) -> List[str]:
    return [c for c in SAFE_LINEUP_CANDIDATES if c in df.columns and c not in LEAKY_LINEUP_COLS]


def prepare_feature_lists(df: pd.DataFrame, savant_feature_set: str = "stable") -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    numeric = [c for c in BASE_NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in BASE_CATEGORICAL_CANDIDATES if c in df.columns]

    savant = stable_savant_features(df) if savant_feature_set == "stable" else all_savant_features(df)
    opp = opponent_features(df)
    env_num, env_cat = environment_features(df)
    lineup = lineup_features(df)

    numeric.extend(savant)
    numeric.extend(opp)
    numeric.extend(env_num)
    numeric.extend(lineup)
    categorical.extend(env_cat)

    numeric = dedupe(numeric)
    categorical = dedupe(categorical)
    numeric = [c for c in numeric if c not in set(categorical)]

    selected = set(numeric) | set(categorical)
    leaky_selected = sorted(selected & LEAKY_LINEUP_COLS)
    if leaky_selected:
        raise RuntimeError(f"Leakage guard triggered. Leaky columns selected: {leaky_selected}")

    meta = {
        "savant": savant,
        "opponent": opp,
        "environment_numeric": env_num,
        "environment_categorical": env_cat,
        "lineup": lineup,
    }
    return numeric, categorical, meta


def drop_empty_features(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> Tuple[List[str], List[str], List[str]]:
    dropped = []
    kept_num = []
    for c in numeric_features:
        if c not in df.columns:
            dropped.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            kept_num.append(c)
        else:
            dropped.append(c)

    kept_cat = []
    for c in categorical_features:
        if c not in df.columns:
            dropped.append(c)
            continue
        s = df[c].astype("string")
        if s.dropna().shape[0] > 0:
            kept_cat.append(c)
        else:
            dropped.append(c)
    return kept_num, kept_cat, sorted(set(dropped))


def build_pipeline(config: ChampionRFConfig, numeric_features: List[str], categorical_features: List[str], random_state: int, n_jobs: int) -> Pipeline:
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
        raise ValueError("No usable features for training")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(**config.params(), random_state=random_state, n_jobs=n_jobs)
    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def feature_importance_frame(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(len(model.feature_importances_))])
    importances = getattr(model, "feature_importances_", np.array([]))
    n = min(len(names), len(importances))
    out = pd.DataFrame({
        "feature": names[:n],
        "importance": importances[:n],
    })
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the QEPC-MLB pitcher strikeouts champion model.")
    parser.add_argument("--input", default="cache/mlb/processed/pitcher_game_logs_enriched_opp_env_lineup.parquet")
    parser.add_argument("--out_dir", default="artifacts/mlb/models/pitcher_strikeouts_champion")
    parser.add_argument("--min_season", type=int, default=2024)
    parser.add_argument("--max_season", type=int, default=2025)
    parser.add_argument("--starters_only", action="store_true")
    parser.add_argument("--min_games", type=int, default=5, help="Minimum prior pitcher games required after rolling features are built.")
    parser.add_argument("--require_savant_rows", action="store_true")
    parser.add_argument("--require_lineup_rows", action="store_true")
    parser.add_argument("--require_env_rows", action="store_true")
    parser.add_argument("--savant_feature_set", choices=["stable", "all"], default="stable")

    # Champion defaults, exposed for controlled future overrides.
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--max_features", default="1.0", help='Use "1.0", "sqrt", "log2", or a float-like string.')
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--also_csv", action="store_true", help="Also save train predictions as CSV.")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def parse_max_features(value: str) -> float | str | None:
    v = str(value).strip().lower()
    if v in {"none", "null"}:
        return None
    if v in {"sqrt", "log2"}:
        return v
    try:
        return float(v)
    except ValueError as exc:
        raise ValueError(f"Invalid --max_features value: {value}") from exc


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading champion training data...")
    df = load_any(args.input)
    raw_rows = int(len(df))
    leaky_found = sorted([c for c in LEAKY_LINEUP_COLS if c in df.columns])

    print("Normalizing schema and applying no-leak guards...")
    df = ensure_core(df)

    print("Building leakage-safe pitcher rolling features...")
    df = compute_pitcher_rolls(df, show_progress=not args.no_progress)

    print("Filtering training universe...")
    df = df[df["game_date"].notna()].copy()
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df["season"].notna()].copy()
    df = df[df["season"] >= args.min_season].copy()
    if args.max_season is not None:
        df = df[df["season"] <= args.max_season].copy()
    if args.starters_only:
        df = df[df["is_starter"].fillna(False).astype(bool)].copy()
    if args.min_games is not None and args.min_games > 0:
        df = df[pd.to_numeric(df.get("games_prior", np.nan), errors="coerce") >= args.min_games].copy()
    if args.require_savant_rows and "has_savant_features" in df.columns:
        df = df[df["has_savant_features"].fillna(False).astype(bool)].copy()
    if args.require_lineup_rows and "has_lineup_context" in df.columns:
        df = df[df["has_lineup_context"].fillna(False).astype(bool)].copy()
    if args.require_env_rows and "has_env_context" in df.columns:
        df = df[df["has_env_context"].fillna(False).astype(bool)].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering. Relax filters or inspect source coverage.")

    numeric_candidates, categorical_candidates, feature_meta = prepare_feature_lists(df, savant_feature_set=args.savant_feature_set)
    numeric_features, categorical_features, dropped_empty = drop_empty_features(df, numeric_candidates, categorical_candidates)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features remain after dropping all-empty columns.")

    X = df[numeric_features + categorical_features].copy()
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_features:
        X[c] = X[c].astype("string")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    config = ChampionRFConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=parse_max_features(args.max_features),
        bootstrap=args.bootstrap,
    )

    print("Training champion model...")
    pipe = build_pipeline(config, numeric_features, categorical_features, random_state=args.random_state, n_jobs=args.n_jobs)
    pipe.fit(X, y)

    print("Generating in-sample predictions for diagnostics...")
    preds = pipe.predict(X)
    train_metrics = metric_dict(y, preds)

    print("Saving artifacts...")
    model_path = out_dir / "model.joblib"
    metadata_path = out_dir / "metadata.json"
    feature_list_path = out_dir / "feature_list.json"
    importance_path = out_dir / "feature_importances.csv"
    pred_path = out_dir / "training_predictions.parquet"

    joblib.dump(pipe, model_path)

    pred_cols = [c for c in ["game_date", "season", "pitcher_id", "pitcher_name", "team", "opponent"] if c in df.columns]
    pred_df = df[pred_cols].copy()
    pred_df["actual_strikeouts"] = y.values
    pred_df["predicted_strikeouts"] = preds
    pred_df["residual_pred_minus_actual"] = pred_df["predicted_strikeouts"] - pred_df["actual_strikeouts"]
    pred_df.to_parquet(pred_path, index=False)
    if args.also_csv:
        pred_df.to_csv(out_dir / "training_predictions.csv", index=False)

    importance_df = feature_importance_frame(pipe)
    importance_df.to_csv(importance_path, index=False)

    feature_payload = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_empty_features": dropped_empty,
        "feature_meta": feature_meta,
    }
    feature_list_path.write_text(json.dumps(feature_payload, indent=2, default=str))

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": str(args.input),
        "raw_rows": raw_rows,
        "rows_after_filtering": int(len(df)),
        "seasons": sorted([int(x) for x in pd.Series(df["season"]).dropna().unique().tolist()]),
        "pitchers": int(df["pitcher_id"].nunique()) if "pitcher_id" in df.columns else None,
        "starter_rows": int(df["is_starter"].fillna(False).astype(bool).sum()) if "is_starter" in df.columns else None,
        "leaky_lineup_columns_found_and_dropped_from_input": leaky_found,
        "globally_excluded_leaky_lineup_features": sorted(LEAKY_LINEUP_COLS),
        "savant_feature_set": args.savant_feature_set,
        "champion_config": config.params(),
        "candidate_numeric_feature_count": int(len(numeric_candidates)),
        "candidate_categorical_feature_count": int(len(categorical_candidates)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "dropped_empty_features": dropped_empty,
        "training_metrics_in_sample": train_metrics,
        "artifact_paths": {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "feature_list": str(feature_list_path),
            "feature_importances": str(importance_path),
            "training_predictions": str(pred_path),
        },
        "note": "In-sample metrics are diagnostics only. Trust the walk-forward tuning/backtest metrics for model selection.",
    }
    metadata_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"saved model: {model_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"saved feature list: {feature_list_path}")
    print(f"saved feature importances: {importance_path}")
    print(f"saved training predictions: {pred_path}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
