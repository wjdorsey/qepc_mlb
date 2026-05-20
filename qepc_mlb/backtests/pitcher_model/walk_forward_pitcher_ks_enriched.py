from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

BASE_NUMERIC = [
    "batters_faced", "innings_pitched", "outs_recorded", "days_rest",
    "walks", "hits_allowed", "earned_runs", "runs_allowed", "games_prior", "target_lag1",
    "k_roll_mean_3", "k_roll_std_3", "k_roll_mean_5", "k_roll_std_5", "k_roll_mean_10", "k_roll_std_10",
    "batters_faced_lag1", "batters_faced_roll5",
    "innings_pitched_lag1", "innings_pitched_roll5",
    "outs_recorded_lag1", "outs_recorded_roll5",
    "days_rest_lag1", "days_rest_roll5",
    "walks_lag1", "walks_roll5",
    "hits_allowed_lag1", "hits_allowed_roll5",
    "earned_runs_lag1", "earned_runs_roll5",
    "runs_allowed_lag1", "runs_allowed_roll5",
    "k_per_bf_lag1", "k_per_bf_roll5",
]
BASE_CATEGORICAL = ["team", "opponent", "pitcher_hand", "is_starter"]
STABLE_SAVANT = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest for QEPC-MLB pitcher strikeouts.")
    p.add_argument("--input", required=True, help="Path to pitcher game logs parquet/csv (baseline or enriched).")
    p.add_argument("--output_dir", default="artifacts/mlb/backtests/pitcher_strikeouts_walk_forward_stable", help="Directory for outputs.")
    p.add_argument("--predictions_out", default="artifacts/mlb/backtests/pitcher_strikeouts_walk_forward_stable/predictions.parquet")
    p.add_argument("--min_season", type=int, default=None)
    p.add_argument("--max_season", type=int, default=None)
    p.add_argument("--starters_only", action="store_true")
    p.add_argument("--min_games", type=int, default=5, help="Minimum appearances per pitcher to keep.")
    p.add_argument("--mode", choices=["baseline", "enriched", "compare"], default="compare")
    p.add_argument("--period", choices=["month", "season"], default="month")
    p.add_argument("--min_train_rows", type=int, default=2000)
    p.add_argument("--require_savant_rows", action="store_true", help="Restrict dataset to rows with has_savant_features=true before backtest.")
    p.add_argument("--savant_feature_set", choices=["stable", "all"], default="stable", help="Which Savant feature set to allow when mode uses enrichment.")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def coerce_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    txt = series.astype("string").str.strip().str.lower()
    out.loc[out.isna() & txt.isin(["true", "t", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & txt.isin(["false", "f", "no", "n", "reliever", "rp"])] = False
    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "is_starter" in out.columns:
        out["is_starter"] = coerce_bool(out["is_starter"]).fillna(False).astype(bool)
    else:
        out["is_starter"] = False
    if "has_savant_features" in out.columns:
        out["has_savant_features"] = coerce_bool(out["has_savant_features"]).fillna(False).astype(bool)
    else:
        out["has_savant_features"] = False
    numeric_candidates = [
        "strikeouts", "batters_faced", "innings_pitched", "outs_recorded", "days_rest",
        "walks", "hits_allowed", "earned_runs", "runs_allowed"
    ] + [c for c in out.columns if c.startswith("savant_")]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["pitcher_id", "game_date"]).copy()
    computed = out.groupby("pitcher_id")["game_date"].diff().dt.days.sub(1)
    if "days_rest" not in out.columns:
        out["days_rest"] = computed
    else:
        out["days_rest"] = out["days_rest"].where(out["days_rest"].notna(), computed)
    return out


def add_rolls(df: pd.DataFrame, show_progress: bool) -> pd.DataFrame:
    out = df.sort_values(["pitcher_id", "game_date"]).copy()
    grouped = out.groupby("pitcher_id", sort=False)
    iterator = grouped if not show_progress else tqdm(grouped, total=out["pitcher_id"].nunique(), desc="Rolling pitcher features")
    pieces = []
    for _, g in iterator:
        g = g.sort_values("game_date").copy()
        g["games_prior"] = np.arange(len(g))
        g["target_lag1"] = g["strikeouts"].shift(1)
        for w in [3, 5, 10]:
            g[f"k_roll_mean_{w}"] = g["strikeouts"].shift(1).rolling(w, min_periods=1).mean()
            g[f"k_roll_std_{w}"] = g["strikeouts"].shift(1).rolling(w, min_periods=2).std()
        for base in ["batters_faced", "innings_pitched", "outs_recorded", "days_rest", "walks", "hits_allowed", "earned_runs", "runs_allowed"]:
            if base in g.columns:
                g[f"{base}_lag1"] = g[base].shift(1)
                g[f"{base}_roll5"] = g[base].shift(1).rolling(5, min_periods=1).mean()
        if "batters_faced" in g.columns:
            bf_lag = g["batters_faced"].shift(1)
            k_lag = g["strikeouts"].shift(1)
            g["k_per_bf_lag1"] = np.where(bf_lag > 0, k_lag / bf_lag, np.nan)
            g["k_per_bf_roll5"] = g["k_per_bf_lag1"].rolling(5, min_periods=1).mean()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def unique_keep_order(cols: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def choose_savant_cols(df: pd.DataFrame, savant_feature_set: str) -> list[str]:
    all_savant = [c for c in df.columns if c.startswith("savant_")]
    if savant_feature_set == "all":
        return [c for c in all_savant if c not in {"savant_team_id", "savant_pa_rv", "savant_season_pitches"}]
    return [c for c in STABLE_SAVANT if c in df.columns]


def build_feature_lists(df: pd.DataFrame, use_savant: bool, savant_feature_set: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    savant_cols = choose_savant_cols(df, savant_feature_set) if use_savant else []
    categorical_candidates = [c for c in BASE_CATEGORICAL if c in df.columns]
    feature_cols = unique_keep_order([c for c in BASE_NUMERIC + savant_cols + categorical_candidates if c in df.columns])

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped_all_empty: list[str] = []
    dropped_non_modelable: list[str] = []

    for c in feature_cols:
        series = df[c]
        if series.notna().sum() == 0:
            dropped_all_empty.append(c)
            continue
        if c in categorical_candidates:
            categorical_cols.append(c)
            continue
        if pd.api.types.is_bool_dtype(series) or str(series.dtype) == "boolean":
            dropped_non_modelable.append(c)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(c)
            continue
        dropped_non_modelable.append(c)

    return numeric_cols, categorical_cols, dropped_all_empty, dropped_non_modelable, savant_cols


def make_period_key(df: pd.DataFrame, period: str) -> pd.Series:
    if period == "season":
        return df["game_date"].dt.year.astype(str)
    return df["game_date"].dt.to_period("M").astype(str)


def make_model(random_state: int, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def eval_metrics(actual: pd.Series, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    bias = float(np.mean(pred - actual))
    return {"rmse": rmse, "mae": mae, "bias": bias}


def run_walk_forward(df: pd.DataFrame, *, use_savant: bool, savant_feature_set: str, period: str, min_train_rows: int,
                     random_state: int, show_progress: bool, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = df.sort_values("game_date").reset_index(drop=True).copy()
    work["period_key"] = make_period_key(work, period)
    periods = sorted(work["period_key"].dropna().unique().tolist())

    candidate_numeric_cols, candidate_categorical_cols, dropped_all_empty_global, dropped_non_modelable_global, savant_candidates = build_feature_lists(
        work, use_savant=use_savant, savant_feature_set=savant_feature_set
    )
    candidate_feature_cols = candidate_numeric_cols + candidate_categorical_cols
    if not candidate_feature_cols:
        raise SystemExit(f"No usable features for mode={label}.")

    per_window = []
    pred_frames = []
    iterator = periods if not show_progress else tqdm(periods, desc=f"Walk-forward {label}")
    for p in iterator:
        test_mask = work["period_key"] == p
        train_mask = work["game_date"] < work.loc[test_mask, "game_date"].min()
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        if len(train) < min_train_rows or test.empty:
            continue

        train_counts = train.groupby("pitcher_id").size()
        keep_ids = set(train_counts[train_counts > 0].index)
        test = test[test["pitcher_id"].isin(keep_ids)].copy()
        if test.empty:
            continue

        numeric_cols = [c for c in candidate_numeric_cols if train[c].notna().sum() > 0]
        categorical_cols = [c for c in candidate_categorical_cols if train[c].notna().sum() > 0]
        dropped_window_empty = [c for c in candidate_feature_cols if c not in numeric_cols + categorical_cols]
        feature_cols = numeric_cols + categorical_cols
        if not feature_cols:
            continue

        X_train = train[feature_cols].copy()
        y_train = train["strikeouts"].astype(float)
        X_test = test[feature_cols].copy()
        y_test = test["strikeouts"].astype(float)

        pipe = make_model(random_state=random_state, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = eval_metrics(y_test, pred)
        window_row = {
            "model_type": label,
            "period_key": p,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": int(len(feature_cols)),
            "numeric_feature_count": int(len(numeric_cols)),
            "categorical_feature_count": int(len(categorical_cols)),
            "dropped_window_empty_count": int(len(dropped_window_empty)),
            "dropped_window_empty_features": "|".join(dropped_window_empty),
            **metrics,
        }
        per_window.append(window_row)

        pf = test[["game_date", "season", "pitcher_id"]].copy()
        if "pitcher_name" in test.columns:
            pf["pitcher_name"] = test["pitcher_name"].values
        pf["model_type"] = label
        pf["period_key"] = p
        pf["actual_strikeouts"] = y_test.values
        pf["predicted_strikeouts"] = pred
        pf["has_savant_features"] = test.get("has_savant_features", pd.Series(False, index=test.index)).fillna(False).astype(bool).values
        pred_frames.append(pf)

    wm = pd.DataFrame(per_window)
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    summary = {
        "model_type": label,
        "savant_feature_set": savant_feature_set if use_savant else "none",
        "windows": int(len(wm)),
        "rows_scored": int(preds.shape[0]),
        "candidate_feature_count": int(len(candidate_feature_cols)),
        "candidate_numeric_feature_count": int(len(candidate_numeric_cols)),
        "candidate_categorical_feature_count": int(len(candidate_categorical_cols)),
        "candidate_savant_features": savant_candidates,
        "globally_empty_features": dropped_all_empty_global,
        "globally_non_modelable_features": dropped_non_modelable_global,
        "avg_feature_count_per_window": None if wm.empty else float(wm["feature_count"].mean()),
        "avg_dropped_window_empty_count": None if wm.empty else float(wm["dropped_window_empty_count"].mean()),
    }
    if not preds.empty:
        overall = eval_metrics(preds["actual_strikeouts"], preds["predicted_strikeouts"].to_numpy())
        weighted_mae = float((wm["mae"] * wm["test_rows"]).sum() / wm["test_rows"].sum()) if not wm.empty else None
        weighted_rmse = float((wm["rmse"] * wm["test_rows"]).sum() / wm["test_rows"].sum()) if not wm.empty else None
        weighted_bias = float((wm["bias"] * wm["test_rows"]).sum() / wm["test_rows"].sum()) if not wm.empty else None
        summary.update({
            "overall_rmse": overall["rmse"],
            "overall_mae": overall["mae"],
            "overall_bias": overall["bias"],
            "weighted_window_mae": weighted_mae,
            "weighted_window_rmse": weighted_rmse,
            "weighted_window_bias": weighted_bias,
            "savant_row_rate_scored": float(preds["has_savant_features"].mean()),
        })
    return wm, preds, summary


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    pred_out = Path(args.predictions_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pitcher game logs...")
    df = load_any(inp)
    print("Normalizing schema...")
    df = normalize(df)
    df = add_days_rest(df)

    if args.min_season is not None:
        df = df[df["season"] >= args.min_season].copy()
    if args.max_season is not None:
        df = df[df["season"] <= args.max_season].copy()
    if args.starters_only:
        df = df[df["is_starter"]].copy()
    if args.require_savant_rows:
        df = df[df["has_savant_features"]].copy()

    df = df[df["game_date"].notna() & df["pitcher_id"].notna() & df["strikeouts"].notna()].copy()
    counts = df.groupby("pitcher_id").size()
    keep_ids = counts[counts >= args.min_games].index
    df = df[df["pitcher_id"].isin(keep_ids)].copy()

    print("Input summary:")
    print(json.dumps({
        "rows": int(len(df)),
        "pitchers": int(df["pitcher_id"].nunique()) if not df.empty else 0,
        "date_min": None if df.empty else str(df["game_date"].min().date()),
        "date_max": None if df.empty else str(df["game_date"].max().date()),
        "starter_rows": int(df["is_starter"].sum()) if "is_starter" in df.columns else 0,
        "has_savant_rows": int(df["has_savant_features"].sum()) if "has_savant_features" in df.columns else 0,
    }, indent=2))
    if df.empty:
        raise SystemExit("No rows available after filtering.")

    print("Building rolling features...")
    df = add_rolls(df, show_progress=not args.no_progress)

    run_labels = []
    if args.mode == "baseline":
        run_labels = [(False, "baseline")]
    elif args.mode == "enriched":
        run_labels = [(True, f"enriched_{args.savant_feature_set}")]
    else:
        run_labels = [(False, "baseline"), (True, f"enriched_{args.savant_feature_set}")]

    wm_parts = []
    pred_parts = []
    summaries = []
    for use_savant, label in run_labels:
        wm, preds, summary = run_walk_forward(
            df,
            use_savant=use_savant,
            savant_feature_set=args.savant_feature_set,
            period=args.period,
            min_train_rows=args.min_train_rows,
            random_state=args.random_state,
            show_progress=not args.no_progress,
            label=label,
        )
        wm_parts.append(wm)
        pred_parts.append(preds)
        summaries.append(summary)

    window_metrics = pd.concat(wm_parts, ignore_index=True) if wm_parts else pd.DataFrame()
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    wm_path = out_dir / "window_metrics.csv"
    summary_path = out_dir / "summary.json"
    window_metrics.to_csv(wm_path, index=False)
    preds.to_parquet(pred_out, index=False)

    payload = {
        "input": str(inp),
        "mode": args.mode,
        "period": args.period,
        "min_season": args.min_season,
        "max_season": args.max_season,
        "starters_only": args.starters_only,
        "require_savant_rows": args.require_savant_rows,
        "savant_feature_set": args.savant_feature_set,
        "summaries": summaries,
    }
    if args.mode == "compare" and len(summaries) == 2:
        base, enr = summaries
        comp = {}
        for k in ["overall_rmse", "overall_mae", "overall_bias", "weighted_window_mae", "weighted_window_rmse", "weighted_window_bias"]:
            if k in base and k in enr:
                comp[k] = {
                    "baseline": base[k],
                    "enriched": enr[k],
                    "delta_enriched_minus_baseline": None if base[k] is None or enr[k] is None else float(enr[k] - base[k]),
                }
        payload["comparison"] = comp
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved predictions: {pred_out}")
    print(f"saved window metrics: {wm_path}")
    print(f"saved summary: {summary_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
