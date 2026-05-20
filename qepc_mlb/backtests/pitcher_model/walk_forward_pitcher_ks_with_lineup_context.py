
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

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
except Exception:
    tqdm = None


TARGET_COL = "strikeouts"
SAFETY_VERSION = "lineup_noleak_v3_assertive"

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


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")


def coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    out.loc[out.isna() & s.isin(["true", "t", "1", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & s.isin(["false", "f", "0", "no", "n", "rp", "reliever"])] = False
    return out


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_date" not in df.columns:
        raise ValueError("Missing game_date")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year
    if "is_starter" in df.columns:
        df["is_starter"] = coerce_bool(df["is_starter"])
    else:
        df["is_starter"] = pd.Series(False, index=df.index, dtype="boolean")
    # normalize numeric-ish core columns
    for c in ["batters_faced", "innings_pitched", "outs_recorded", "days_rest", "walks", "hits_allowed",
              "earned_runs", "runs_allowed", TARGET_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_pitcher_rolls(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    df = df.copy()
    if "pitcher_id" not in df.columns:
        raise ValueError("Missing pitcher_id")
    df = df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    groups = df.groupby("pitcher_id", sort=False)

    df["games_prior"] = groups.cumcount()
    df["target_lag1"] = groups[TARGET_COL].shift(1)

    for window in [3, 5, 10]:
        df[f"k_roll_mean_{window}"] = groups[TARGET_COL].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        df[f"k_roll_std_{window}"] = groups[TARGET_COL].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())

    for base in ["batters_faced", "innings_pitched", "outs_recorded", "days_rest", "walks", "hits_allowed", "earned_runs", "runs_allowed"]:
        if base in df.columns:
            df[f"{base}_lag1"] = groups[base].shift(1)
            df[f"{base}_roll5"] = groups[base].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    if "batters_faced" in df.columns:
        bf_prior = groups["batters_faced"].shift(1)
        k_prior = groups[TARGET_COL].shift(1)
        df["k_per_bf_lag1"] = k_prior / bf_prior.replace(0, np.nan)
        df["k_per_bf_roll5"] = groups.apply(
            lambda g: (g[TARGET_COL].shift(1).rolling(5, min_periods=1).sum() /
                       g["batters_faced"].shift(1).rolling(5, min_periods=1).sum().replace(0, np.nan))
        ).reset_index(level=0, drop=True)
    return df


def stable_savant_features(df: pd.DataFrame) -> List[str]:
    candidates = [
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
    disallow = {"player_id", "pitcher_name_savant"}
    return [c for c in df.columns if c.startswith("savant_") and c not in disallow]


def opponent_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("opp_team_")]


def environment_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num, cat = [], []
    for c in df.columns:
        if c.startswith("env_"):
            if c in {"env_winddir", "env_daynight", "env_park_id"}:
                cat.append(c)
            else:
                num.append(c)
    if "has_env_context" in df.columns:
        num.append("has_env_context")
    return num, cat


def lineup_features(df: pd.DataFrame) -> List[str]:
    return [c for c in SAFE_LINEUP_CANDIDATES if c in df.columns and c not in LEAKY_LINEUP_COLS]


def prepare_feature_lists(df: pd.DataFrame, use_enrichment: bool, savant_feature_set: str) -> Tuple[List[str], List[str], List[str]]:
    numeric = [c for c in BASE_NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in BASE_CATEGORICAL_CANDIDATES if c in df.columns]

    candidate_meta = []
    if use_enrichment:
        if savant_feature_set == "stable":
            numeric.extend(stable_savant_features(df))
        else:
            numeric.extend(all_savant_features(df))
        numeric.extend(opponent_features(df))
        env_num, env_cat = environment_features(df)
        numeric.extend(env_num)
        categorical.extend(env_cat)
        numeric.extend(lineup_features(df))

    # coerce potential bools/objects in numeric candidates later; for now dedupe preserving order
    seen = set()
    numeric = [c for c in numeric if not (c in seen or seen.add(c))]
    seen = set()
    categorical = [c for c in categorical if not (c in seen or seen.add(c))]
    # remove categorical from numeric just in case
    numeric = [c for c in numeric if c not in set(categorical)]
    all_features = numeric + categorical
    leaky_selected = sorted(set(all_features) & LEAKY_LINEUP_COLS)
    if leaky_selected:
        raise RuntimeError(f"Leakage guard triggered: leaky lineup columns selected: {leaky_selected}")
    return all_features, numeric, categorical


def build_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]),
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

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def month_periods(df: pd.DataFrame) -> List[pd.Period]:
    return sorted(df["game_date"].dt.to_period("M").dropna().unique().tolist())


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def run_walk_forward(
    df: pd.DataFrame,
    use_enrichment: bool,
    savant_feature_set: str,
    min_train_rows: int,
    random_state: int,
    show_progress: bool,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    periods = month_periods(df)
    iterator = periods[1:]
    if tqdm is not None and show_progress:
        iterator = tqdm(iterator, desc=f"Walk-forward {label}")

    window_rows = []
    preds_parts = []

    for period in iterator:
        test_mask = df["game_date"].dt.to_period("M") == period
        train_mask = df["game_date"] < period.start_time
        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()
        if len(train) < min_train_rows or test.empty:
            continue

        all_features, numeric_features, categorical_features = prepare_feature_lists(train, use_enrichment, savant_feature_set)

        # Drop window-empty features using training slice only
        dropped_empty = []
        kept_num = []
        for c in numeric_features:
            s = pd.to_numeric(train[c], errors="coerce") if c in train.columns else pd.Series(dtype=float)
            if s.notna().any():
                kept_num.append(c)
            else:
                dropped_empty.append(c)

        kept_cat = []
        for c in categorical_features:
            s = train[c].astype("string") if c in train.columns else pd.Series(dtype="string")
            if s.dropna().shape[0] > 0:
                kept_cat.append(c)
            else:
                dropped_empty.append(c)

        if not kept_num and not kept_cat:
            continue

        X_train = train[kept_num + kept_cat].copy()
        X_test = test[kept_num + kept_cat].copy()
        for c in kept_num:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        for c in kept_cat:
            X_train[c] = X_train[c].astype("string")
            X_test[c] = X_test[c].astype("string")

        y_train = pd.to_numeric(train[TARGET_COL], errors="coerce")
        y_test = pd.to_numeric(test[TARGET_COL], errors="coerce")

        pipe = build_pipeline(kept_num, kept_cat, random_state)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        m = metric_dict(y_test, preds)
        row = {
            "label": label,
            "window": str(period),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": int(len(kept_num) + len(kept_cat)),
            "numeric_feature_count": int(len(kept_num)),
            "categorical_feature_count": int(len(kept_cat)),
            "dropped_window_empty_count": int(len(dropped_empty)),
            "dropped_window_empty_features": ",".join(sorted(dropped_empty)),
            **m,
        }
        window_rows.append(row)

        part = test[["game_date", "pitcher_id", "pitcher_name"]].copy() if "pitcher_name" in test.columns else test[["game_date", "pitcher_id"]].copy()
        part["window"] = str(period)
        part["label"] = label
        part["actual"] = y_test.values
        part["pred"] = preds
        preds_parts.append(part)

    wm = pd.DataFrame(window_rows)
    preds = pd.concat(preds_parts, ignore_index=True) if preds_parts else pd.DataFrame()

    if preds.empty:
        summary = {"overall_rmse": np.nan, "overall_mae": np.nan, "overall_bias": np.nan}
    else:
        overall = metric_dict(preds["actual"], preds["pred"])
        w = wm["test_rows"].astype(float)
        summary = {
            "overall_rmse": overall["rmse"],
            "overall_mae": overall["mae"],
            "overall_bias": overall["bias"],
            "weighted_window_mae": float(np.average(wm["mae"], weights=w)),
            "weighted_window_rmse": float(np.average(wm["rmse"], weights=w)),
            "weighted_window_bias": float(np.average(wm["bias"], weights=w)),
            "avg_feature_count_per_window": float(wm["feature_count"].mean()),
            "avg_dropped_window_empty_count": float(wm["dropped_window_empty_count"].mean()),
        }
    return wm, preds, summary


def compare_summaries(base: Dict[str, float], enr: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in ["overall_rmse", "overall_mae", "overall_bias", "weighted_window_mae", "weighted_window_rmse", "weighted_window_bias"]:
        if k in base and k in enr:
            out[k] = {
                "baseline": base[k],
                "enriched": enr[k],
                "delta_enriched_minus_baseline": enr[k] - base[k],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe walk-forward pitcher K backtest with lineup context.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="artifacts/mlb/backtests/pitcher_strikeouts_walk_forward_lineup_noleak")
    ap.add_argument("--mode", choices=["baseline", "enriched", "compare"], default="compare")
    ap.add_argument("--min_season", type=int, default=None)
    ap.add_argument("--max_season", type=int, default=None)
    ap.add_argument("--starters_only", action="store_true")
    ap.add_argument("--min_games", type=int, default=5)
    ap.add_argument("--require_savant_rows", action="store_true")
    ap.add_argument("--require_lineup_rows", action="store_true")
    ap.add_argument("--savant_feature_set", choices=["stable", "all"], default="stable")
    ap.add_argument("--min_train_rows", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    df = load_any(args.input)
    df = ensure_core(df)
    df = compute_pitcher_rolls(df, show_progress=not args.no_progress)

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

    df = df[df["games_prior"] >= args.min_games].copy()
    df = df.dropna(subset=["game_date", TARGET_COL, "pitcher_id"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaked_present = sorted([c for c in LEAKY_LINEUP_COLS if c in df.columns])
    if leaked_present:
        df = df.drop(columns=leaked_present)

    summary = {
        "safety_version": SAFETY_VERSION,
        "rows": int(len(df)),
        "seasons": sorted(pd.Series(df["season"]).dropna().astype(int).unique().tolist()),
        "globally_excluded_leaky_lineup_features": sorted(LEAKY_LINEUP_COLS),
        "leaky_lineup_columns_found_and_dropped_from_input": leaked_present,
        "safe_lineup_candidate_features": [c for c in SAFE_LINEUP_CANDIDATES if c in df.columns],
    }

    all_wm = []
    all_preds = []

    if args.mode in {"baseline", "compare"}:
        wm_base, preds_base, summ_base = run_walk_forward(
            df, use_enrichment=False, savant_feature_set=args.savant_feature_set,
            min_train_rows=args.min_train_rows, random_state=args.random_state,
            show_progress=not args.no_progress, label="baseline",
        )
        all_wm.append(wm_base)
        all_preds.append(preds_base)
        summary["baseline"] = summ_base

    if args.mode in {"enriched", "compare"}:
        wm_enr, preds_enr, summ_enr = run_walk_forward(
            df, use_enrichment=True, savant_feature_set=args.savant_feature_set,
            min_train_rows=args.min_train_rows, random_state=args.random_state,
            show_progress=not args.no_progress, label="enriched",
        )
        all_wm.append(wm_enr)
        all_preds.append(preds_enr)
        summary["enriched"] = summ_enr

    if args.mode == "compare" and "baseline" in summary and "enriched" in summary:
        summary["comparison"] = compare_summaries(summary["baseline"], summary["enriched"])

    wm = pd.concat(all_wm, ignore_index=True) if all_wm else pd.DataFrame()
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    wm_path = out_dir / "window_metrics.csv"
    preds_path = out_dir / "predictions.parquet"
    summ_path = out_dir / "summary.json"
    wm.to_csv(wm_path, index=False)
    if not preds.empty:
        preds.to_parquet(preds_path, index=False)
    summ_path.write_text(json.dumps(summary, indent=2))

    print(f"saved window metrics: {wm_path}")
    if not preds.empty:
        print(f"saved predictions: {preds_path}")
    print(f"saved summary: {summ_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
