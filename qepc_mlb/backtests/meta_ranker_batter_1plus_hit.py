#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Meta-Ranker / Resolver v1.

This is a second-stage model that learns from existing model outputs:

  - v3 probability
  - Savant-light probability
  - blend probability
  - daily ranks/percentiles
  - model disagreement features

Safety:
  The meta model is walk-forward trained.
  For each test period, it only trains on earlier prediction periods.
  Base predictions should be out-of-fold / walk-forward predictions from saved tuner runs.

Example:

  python qepc_mlb/backtests/meta_ranker_batter_1plus_hit.py \
    --v3_preds artifacts/mlb/backtests/blend_source_v3_monthly/predictions.parquet \
    --savant_preds artifacts/mlb/backtests/blend_source_savant_light_monthly/predictions.parquet \
    --out_dir artifacts/mlb/backtests/batter_1plus_hit_meta_ranker_v1 \
    --period M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SAFETY_VERSION = "batter_1plus_hit_meta_ranker_v1_walk_forward_resolver"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward meta-ranker for Batter 1+ Hit predictions")

    p.add_argument("--v3_preds", required=True)
    p.add_argument("--savant_preds", required=True)
    p.add_argument("--out_dir", default="artifacts/mlb/backtests/batter_1plus_hit_meta_ranker_v1")

    p.add_argument("--period", choices=["M", "Q"], default="M")
    p.add_argument("--min_train_periods", type=int, default=4)
    p.add_argument("--min_train_rows", type=int, default=5000)

    p.add_argument("--v3_weight", type=float, default=0.80)
    p.add_argument("--savant_weight", type=float, default=0.20)

    p.add_argument("--model", choices=["hgb", "logit", "both"], default="both")
    p.add_argument("--random_state", type=int, default=42)

    return p.parse_args()


def read_preds(path: str | Path, prob_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported prediction file: {path}")

    required = ["game_date", "batter_id", "team", "opponent", "lineup_slot", "actual", "pred_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if "game_id" not in df.columns:
        raise ValueError(
            f"{path} is missing game_id. Re-run the tuner after patching prediction outputs to include game_id."
        )

    out = df.copy()
    out["game_id"] = out["game_id"].astype("string")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["batter_id"] = out["batter_id"].astype("string")
    out["team"] = out["team"].astype("string")
    out["opponent"] = out["opponent"].astype("string")
    out["lineup_slot"] = pd.to_numeric(out["lineup_slot"], errors="coerce")
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce").astype(int)
    out["pred_prob"] = pd.to_numeric(out["pred_prob"], errors="coerce")

    keep = [
        "game_id",
        "game_date",
        "batter_id",
        "batter_name",
        "team",
        "opponent",
        "lineup_slot",
        "actual",
        "pred_prob",
    ]
    keep = [c for c in keep if c in out.columns]

    out = out[keep].rename(columns={"pred_prob": prob_name, "actual": f"actual_{prob_name}"})

    if "batter_name" in out.columns:
        out = out.rename(columns={"batter_name": f"batter_name_{prob_name}"})

    return out


def merge_predictions(v3: pd.DataFrame, sav: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "batter_id", "team", "opponent", "lineup_slot"]

    v3_dupes = int(v3.duplicated(keys).sum())
    sav_dupes = int(sav.duplicated(keys).sum())

    if v3_dupes:
        raise ValueError(f"v3 predictions have duplicate merge keys: {v3_dupes}")
    if sav_dupes:
        raise ValueError(f"savant predictions have duplicate merge keys: {sav_dupes}")

    m = v3.merge(sav, how="inner", on=keys, suffixes=("_v3", "_sav"), validate="one_to_one")

    # Choose one date/name.
    if "game_date_v3" in m.columns:
        m["game_date"] = pd.to_datetime(m["game_date_v3"], errors="coerce")
    elif "game_date" in m.columns:
        m["game_date"] = pd.to_datetime(m["game_date"], errors="coerce")
    else:
        raise ValueError("Merged predictions missing game_date.")

    if "batter_name_v3_prob" in m.columns:
        m["batter_name"] = m["batter_name_v3_prob"]
    elif "batter_name_savant_prob" in m.columns:
        m["batter_name"] = m["batter_name_savant_prob"]
    elif "batter_name_v3" in m.columns:
        m["batter_name"] = m["batter_name_v3"]
    else:
        m["batter_name"] = pd.NA

    actual_v3_col = "actual_v3_prob"
    actual_sav_col = "actual_savant_prob"

    if actual_v3_col not in m.columns or actual_sav_col not in m.columns:
        raise ValueError(f"Missing actual columns after merge. Columns: {list(m.columns)}")

    agreement = (m[actual_v3_col].astype(int) == m[actual_sav_col].astype(int)).mean()
    if agreement < 0.999999:
        raise ValueError(f"Actual label agreement below 1.0: {agreement}")

    m["actual"] = m[actual_v3_col].astype(int)

    return m


def add_meta_features(df: pd.DataFrame, v3_weight: float, savant_weight: float) -> pd.DataFrame:
    df = df.copy()

    df["v3_prob"] = pd.to_numeric(df["v3_prob"], errors="coerce")
    df["savant_prob"] = pd.to_numeric(df["savant_prob"], errors="coerce")

    df["blend_80_20_prob"] = (v3_weight * df["v3_prob"]) + (savant_weight * df["savant_prob"])
    df["prob_mean"] = 0.5 * (df["v3_prob"] + df["savant_prob"])
    df["prob_max"] = df[["v3_prob", "savant_prob"]].max(axis=1)
    df["prob_min"] = df[["v3_prob", "savant_prob"]].min(axis=1)
    df["prob_diff_savant_minus_v3"] = df["savant_prob"] - df["v3_prob"]
    df["prob_abs_diff"] = (df["savant_prob"] - df["v3_prob"]).abs()
    df["models_agree_close_01"] = (df["prob_abs_diff"] <= 0.01).astype(int)
    df["models_agree_close_025"] = (df["prob_abs_diff"] <= 0.025).astype(int)
    df["models_agree_close_05"] = (df["prob_abs_diff"] <= 0.05).astype(int)
    df["savant_above_v3"] = (df["savant_prob"] > df["v3_prob"]).astype(int)

    # Daily ranks and percentiles. Rank 1 = best.
    for col in ["v3_prob", "savant_prob", "blend_80_20_prob", "prob_mean", "prob_max"]:
        rank_col = col.replace("_prob", "") + "_rank_daily"
        pct_col = col.replace("_prob", "") + "_pct_daily"

        df[rank_col] = df.groupby("game_date")[col].rank(method="first", ascending=False)
        df[pct_col] = df.groupby("game_date")[col].rank(method="average", pct=True)

    df["rank_diff_savant_minus_v3"] = df["savant_rank_daily"] - df["v3_rank_daily"]
    df["rank_abs_diff"] = df["rank_diff_savant_minus_v3"].abs()

    # Candidate region indicators.
    for k in [5, 10, 25, 50, 75, 100]:
        df[f"v3_top{k}"] = (df["v3_rank_daily"] <= k).astype(int)
        df[f"savant_top{k}"] = (df["savant_rank_daily"] <= k).astype(int)
        df[f"both_top{k}"] = ((df["v3_rank_daily"] <= k) & (df["savant_rank_daily"] <= k)).astype(int)
        df[f"either_top{k}"] = ((df["v3_rank_daily"] <= k) | (df["savant_rank_daily"] <= k)).astype(int)

    df["lineup_slot_num"] = pd.to_numeric(df["lineup_slot"], errors="coerce")
    df["lineup_top3"] = df["lineup_slot_num"].isin([1, 2, 3]).astype(int)
    df["lineup_top5"] = df["lineup_slot_num"].isin([1, 2, 3, 4, 5]).astype(int)

    return df


FEATURES = [
    "v3_prob",
    "savant_prob",
    "blend_80_20_prob",
    "prob_mean",
    "prob_max",
    "prob_min",
    "prob_diff_savant_minus_v3",
    "prob_abs_diff",
    "models_agree_close_01",
    "models_agree_close_025",
    "models_agree_close_05",
    "savant_above_v3",
    "v3_rank_daily",
    "savant_rank_daily",
    "blend_80_20_rank_daily",
    "prob_mean_rank_daily",
    "prob_max_rank_daily",
    "v3_pct_daily",
    "savant_pct_daily",
    "blend_80_20_pct_daily",
    "prob_mean_pct_daily",
    "prob_max_pct_daily",
    "rank_diff_savant_minus_v3",
    "rank_abs_diff",
    "lineup_slot_num",
    "lineup_top3",
    "lineup_top5",
    "v3_top5",
    "savant_top5",
    "both_top5",
    "either_top5",
    "v3_top10",
    "savant_top10",
    "both_top10",
    "either_top10",
    "v3_top25",
    "savant_top25",
    "both_top25",
    "either_top25",
    "v3_top50",
    "savant_top50",
    "both_top50",
    "either_top50",
    "v3_top75",
    "savant_top75",
    "both_top75",
    "either_top75",
    "v3_top100",
    "savant_top100",
    "both_top100",
    "either_top100",
]


def make_model(model_name: str, random_state: int):
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=random_state,
        )

    if model_name == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
            ),
        )

    raise ValueError(model_name)


def get_periods(df: pd.DataFrame, period: str) -> pd.Series:
    return df["game_date"].dt.to_period(period).astype(str)


def topk_daily(df: pd.DataFrame, prob_col: str, k: int) -> Dict[str, Any]:
    top = (
        df.sort_values(["game_date", prob_col], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(k)
    )

    return {
        f"top{k}_daily_hit_rate": float(top["actual"].mean()) if len(top) else None,
        f"top{k}_daily_dates": int(top["game_date"].nunique()) if len(top) else 0,
        f"top{k}_daily_rows": int(len(top)),
    }


def evaluate(df: pd.DataFrame, prob_col: str) -> Dict[str, Any]:
    y = df["actual"].astype(int).to_numpy()
    p = df[prob_col].clip(1e-6, 1 - 1e-6).to_numpy()

    out = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "actual_rate": float(np.mean(y)),
        "avg_pred_prob": float(np.mean(p)),
        "prob_bias": float(np.mean(p) - np.mean(y)),
        "rows": int(len(df)),
        "dates": int(df["game_date"].nunique()),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = None

    try:
        out["avg_precision"] = float(average_precision_score(y, p))
    except Exception:
        out["avg_precision"] = None

    for k in [5, 10, 25, 50]:
        out.update(topk_daily(df, prob_col, k))

    return out


def walk_forward_meta(
    df: pd.DataFrame,
    model_name: str,
    period: str,
    min_train_periods: int,
    min_train_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["_period"] = get_periods(df, period)

    periods = sorted(df["_period"].dropna().unique().tolist())

    preds = []
    window_rows = []

    for i, p in enumerate(periods):
        train_periods = periods[:i]

        if len(train_periods) < min_train_periods:
            continue

        train = df[df["_period"].isin(train_periods)].copy()
        test = df[df["_period"].eq(p)].copy()

        if len(train) < min_train_rows or test.empty:
            continue

        x_train = train[FEATURES].copy()
        y_train = train["actual"].astype(int)
        x_test = test[FEATURES].copy()

        model = make_model(model_name, random_state=random_state)
        model.fit(x_train, y_train)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test)[:, 1]
        else:
            prob = model.predict(x_test)

        pred = test[
            [
                "game_id",
                "game_date",
                "batter_id",
                "batter_name",
                "team",
                "opponent",
                "lineup_slot",
                "actual",
                "v3_prob",
                "savant_prob",
                "blend_80_20_prob",
            ]
        ].copy()

        pred["meta_prob"] = prob
        pred["model"] = model_name
        pred["period"] = p
        preds.append(pred)

        metrics = evaluate(pred.rename(columns={"meta_prob": "prob"}), "prob")
        metrics["model"] = model_name
        metrics["period"] = p
        metrics["train_rows"] = int(len(train))
        metrics["test_rows"] = int(len(test))
        window_rows.append(metrics)

    if preds:
        pred_df = pd.concat(preds, ignore_index=True)
    else:
        pred_df = pd.DataFrame()

    window_df = pd.DataFrame(window_rows)
    return pred_df, window_df


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v3 = read_preds(args.v3_preds, "v3_prob")
    sav = read_preds(args.savant_preds, "savant_prob")

    merged = merge_predictions(v3, sav)
    merged = add_meta_features(merged, args.v3_weight, args.savant_weight)

    model_names = ["hgb", "logit"] if args.model == "both" else [args.model]

    all_preds = []
    all_windows = []
    summaries = []

    for model_name in model_names:
        pred_df, window_df = walk_forward_meta(
            merged,
            model_name=model_name,
            period=args.period,
            min_train_periods=args.min_train_periods,
            min_train_rows=args.min_train_rows,
            random_state=args.random_state,
        )

        if pred_df.empty:
            continue

        metrics = evaluate(pred_df, "meta_prob")
        metrics["model"] = model_name
        metrics["windows"] = int(pred_df["period"].nunique())
        metrics["rows_scored"] = int(len(pred_df))
        summaries.append(metrics)

        all_preds.append(pred_df)
        all_windows.append(window_df)

    if not summaries:
        raise SystemExit("No meta-ranker predictions were produced. Try lowering --min_train_periods or --min_train_rows.")

    summary_rank = pd.DataFrame(summaries).sort_values(
        ["top10_daily_hit_rate", "top25_daily_hit_rate", "brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_model = str(summary_rank.iloc[0]["model"])

    preds = pd.concat(all_preds, ignore_index=True)
    windows = pd.concat(all_windows, ignore_index=True)

    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    windows.to_csv(out_dir / "window_metrics.csv", index=False)
    summary_rank.to_csv(out_dir / "model_ranking.csv", index=False)

    best_preds = preds[preds["model"].eq(best_model)].copy()
    best_top25 = (
        best_preds.sort_values(["game_date", "meta_prob"], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(25)
        .copy()
    )
    best_top25.to_csv(out_dir / "best_meta_top25_daily_board.csv", index=False)

    # Also evaluate baseline v3/blend on the SAME scored rows for fair comparison.
    baseline_rows = merged.merge(
        best_preds[["game_id", "batter_id", "team", "opponent", "lineup_slot"]],
        how="inner",
        on=["game_id", "batter_id", "team", "opponent", "lineup_slot"],
        validate="one_to_one",
    )

    baseline_v3 = evaluate(baseline_rows, "v3_prob")
    baseline_savant = evaluate(baseline_rows, "savant_prob")
    baseline_blend = evaluate(baseline_rows, "blend_80_20_prob")

    summary = {
        "safety_version": SAFETY_VERSION,
        "v3_preds": str(args.v3_preds),
        "savant_preds": str(args.savant_preds),
        "rows_merged": int(len(merged)),
        "dates_merged": int(merged["game_date"].nunique()),
        "period": args.period,
        "min_train_periods": args.min_train_periods,
        "min_train_rows": args.min_train_rows,
        "features": FEATURES,
        "best_model": summary_rank.iloc[0].to_dict(),
        "model_ranking": summary_rank.to_dict(orient="records"),
        "same_rows_baselines": {
            "v3": baseline_v3,
            "savant": baseline_savant,
            "blend_80_20": baseline_blend,
        },
        "outputs": {
            "predictions": str(out_dir / "predictions.parquet"),
            "window_metrics": str(out_dir / "window_metrics.csv"),
            "model_ranking": str(out_dir / "model_ranking.csv"),
            "best_meta_top25_daily_board": str(out_dir / "best_meta_top25_daily_board.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps({
        "best_model": summary["best_model"],
        "model_ranking": summary["model_ranking"],
        "same_rows_baselines": summary["same_rows_baselines"],
        "outputs": summary["outputs"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
