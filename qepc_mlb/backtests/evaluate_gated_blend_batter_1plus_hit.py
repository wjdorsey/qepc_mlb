#!/usr/bin/env python3
"""
Evaluate rule-based gated blends for QEPC-MLB Batter 1+ Hit board.

Uses saved walk-forward predictions from:
  - v3
  - Savant-light challenger

This is NOT a trained model. It tests simple QEPC-style collapse rules:
  - agreement boost
  - disagreement penalty
  - v3 top-5 protection
  - Savant upgrade gates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


SAFETY_VERSION = "batter_1plus_hit_gated_blend_eval_v1_gameid_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v3_preds", required=True)
    p.add_argument("--savant_preds", required=True)
    p.add_argument("--out_dir", default="artifacts/mlb/backtests/batter_1plus_hit_gated_blend_v1")
    return p.parse_args()


def read_preds(path: str | Path, prob_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = ["game_id", "game_date", "batter_id", "team", "opponent", "lineup_slot", "actual", "pred_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.copy()
    df["game_id"] = df["game_id"].astype("string")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["batter_id"] = df["batter_id"].astype("string")
    df["team"] = df["team"].astype("string")
    df["opponent"] = df["opponent"].astype("string")
    df["lineup_slot"] = pd.to_numeric(df["lineup_slot"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")

    keep = ["game_id", "game_date", "batter_id", "batter_name", "team", "opponent", "lineup_slot", "actual", "pred_prob"]
    keep = [c for c in keep if c in df.columns]

    return df[keep].rename(columns={"actual": f"actual_{prob_name}", "pred_prob": prob_name})


def merge_sources(v3: pd.DataFrame, sav: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "batter_id", "team", "opponent", "lineup_slot"]

    print("v3 rows:", len(v3))
    print("savant rows:", len(sav))
    print("v3 duplicate keys:", int(v3.duplicated(keys).sum()))
    print("savant duplicate keys:", int(sav.duplicated(keys).sum()))

    v3_small = v3.rename(columns={"batter_name": "batter_name_v3"})
    sav_small = sav.rename(columns={"batter_name": "batter_name_savant"})

    m = v3_small.merge(sav_small, how="inner", on=keys, suffixes=("_v3", "_sav"), validate="one_to_one")

    # Date should agree, but use v3 date as canonical.
    if "game_date_v3" in m.columns:
        m["game_date"] = m["game_date_v3"]
    elif "game_date" not in m.columns:
        raise ValueError("Missing game_date after merge")

    m["batter_name"] = m.get("batter_name_v3", pd.Series(pd.NA, index=m.index)).fillna(
        m.get("batter_name_savant", pd.Series(pd.NA, index=m.index))
    )

    agree = (m["actual_v3_prob"].astype(int) == m["actual_savant_prob"].astype(int)).mean()
    print("rows merged:", len(m))
    print("actual agreement:", agree)

    if agree < 0.999999:
        raise ValueError(f"Actual labels disagree: {agree}")

    m["actual"] = m["actual_v3_prob"].astype(int)

    return m


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["base_80_20"] = 0.80 * df["v3_prob"] + 0.20 * df["savant_prob"]
    df["base_85_15"] = 0.85 * df["v3_prob"] + 0.15 * df["savant_prob"]
    df["base_75_25"] = 0.75 * df["v3_prob"] + 0.25 * df["savant_prob"]

    df["prob_diff"] = df["savant_prob"] - df["v3_prob"]
    df["prob_abs_diff"] = df["prob_diff"].abs()

    for col in ["v3_prob", "savant_prob", "base_80_20", "base_85_15", "base_75_25"]:
        short = col.replace("_prob", "")
        df[f"{short}_rank_daily"] = df.groupby("game_date")[col].rank(method="first", ascending=False)

    df["both_top10"] = ((df["v3_rank_daily"] <= 10) & (df["savant_rank_daily"] <= 10)).astype(int)
    df["both_top25"] = ((df["v3_rank_daily"] <= 25) & (df["savant_rank_daily"] <= 25)).astype(int)
    df["both_top50"] = ((df["v3_rank_daily"] <= 50) & (df["savant_rank_daily"] <= 50)).astype(int)

    df["either_top25"] = ((df["v3_rank_daily"] <= 25) | (df["savant_rank_daily"] <= 25)).astype(int)

    return df


def clip_prob(x):
    return np.clip(x, 1e-6, 1 - 1e-6)


def add_gated_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Baselines.
    df["score_v3"] = df["v3_prob"]
    df["score_savant"] = df["savant_prob"]
    df["score_blend_80_20"] = df["base_80_20"]
    df["score_blend_85_15"] = df["base_85_15"]
    df["score_blend_75_25"] = df["base_75_25"]

    # Gate A: protect v3's top-5; otherwise use 80/20.
    df["score_gate_v3_top5_protect"] = np.where(
        df["v3_rank_daily"] <= 5,
        0.90 * df["v3_prob"] + 0.10 * df["savant_prob"],
        df["base_80_20"],
    )

    # Gate B: if both models agree on top candidates, small boost.
    df["score_gate_agreement_boost"] = df["base_80_20"]
    df.loc[df["both_top10"].eq(1), "score_gate_agreement_boost"] += 0.008
    df.loc[df["both_top25"].eq(1), "score_gate_agreement_boost"] += 0.004
    df.loc[df["prob_abs_diff"] > 0.06, "score_gate_agreement_boost"] -= 0.004
    df["score_gate_agreement_boost"] = clip_prob(df["score_gate_agreement_boost"])

    # Gate C: dynamic weighting.
    # - v3 top 5: stay v3-heavy
    # - both top 25: lean slightly more Savant
    # - Savant strongly upgrades a player inside top 50: allow more Savant
    # - hard disagreement outside v3 top 25: penalize
    score = df["base_80_20"].copy()

    v3_top5 = df["v3_rank_daily"] <= 5
    both25 = df["both_top25"].eq(1)
    savant_upgrade = (df["prob_diff"] >= 0.025) & (df["savant_rank_daily"] <= 50)
    hard_disagree = (df["prob_abs_diff"] >= 0.075) & (df["v3_rank_daily"] > 25)

    score = np.where(v3_top5, 0.90 * df["v3_prob"] + 0.10 * df["savant_prob"], score)
    score = np.where((~v3_top5) & both25, 0.75 * df["v3_prob"] + 0.25 * df["savant_prob"], score)
    score = np.where((~v3_top5) & (~both25) & savant_upgrade, 0.65 * df["v3_prob"] + 0.35 * df["savant_prob"], score)
    score = np.where(hard_disagree, score - 0.005, score)

    df["score_gate_dynamic_a"] = clip_prob(score)

    # Gate D: "collapse consensus" score: blend + rank agreement bonus.
    df["score_collapse_consensus"] = df["base_80_20"]
    df.loc[df["both_top10"].eq(1), "score_collapse_consensus"] += 0.010
    df.loc[(df["both_top25"].eq(1)) & (df["both_top10"].eq(0)), "score_collapse_consensus"] += 0.006
    df.loc[(df["both_top50"].eq(1)) & (df["both_top25"].eq(0)), "score_collapse_consensus"] += 0.002
    df.loc[(df["prob_abs_diff"] >= 0.06) & (df["both_top25"].eq(0)), "score_collapse_consensus"] -= 0.004
    df["score_collapse_consensus"] = clip_prob(df["score_collapse_consensus"])

    # Gate E: top-board specialist: deliberately cares about ranking more than calibration.
    df["score_topboard_specialist"] = df["base_80_20"]
    df.loc[df["both_top10"].eq(1), "score_topboard_specialist"] += 0.012
    df.loc[df["both_top25"].eq(1), "score_topboard_specialist"] += 0.006
    df.loc[(df["v3_rank_daily"] <= 5) & (df["savant_rank_daily"] <= 25), "score_topboard_specialist"] += 0.004
    df.loc[(df["savant_rank_daily"] <= 10) & (df["prob_diff"] >= 0.02), "score_topboard_specialist"] += 0.004
    df.loc[(df["prob_abs_diff"] >= 0.08) & (df["both_top25"].eq(0)), "score_topboard_specialist"] -= 0.006
    df["score_topboard_specialist"] = clip_prob(df["score_topboard_specialist"])

    return df


def topk_daily(df: pd.DataFrame, score_col: str, k: int) -> Dict[str, Any]:
    top = (
        df.sort_values(["game_date", score_col], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(k)
    )

    return {
        f"top{k}_daily_hit_rate": float(top["actual"].mean()) if len(top) else None,
        f"top{k}_daily_dates": int(top["game_date"].nunique()) if len(top) else 0,
        f"top{k}_daily_rows": int(len(top)),
    }


def metrics_for(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    y = df["actual"].to_numpy()
    p = clip_prob(df[score_col].to_numpy())

    out = {
        "score": score_col,
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
        out.update(topk_daily(df, score_col, k))

    return out


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v3 = read_preds(args.v3_preds, "v3_prob")
    sav = read_preds(args.savant_preds, "savant_prob")

    df = merge_sources(v3, sav)
    df = add_daily_features(df)
    df = add_gated_scores(df)

    score_cols = [c for c in df.columns if c.startswith("score_")]
    results = [metrics_for(df, c) for c in score_cols]

    rank = pd.DataFrame(results).sort_values(
        ["top10_daily_hit_rate", "top25_daily_hit_rate", "top5_daily_hit_rate", "brier"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    best = rank.iloc[0].to_dict()
    best_score = best["score"]

    top25 = (
        df.sort_values(["game_date", best_score], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(25)
        .copy()
    )

    top25 = top25[
        [
            "game_id",
            "game_date",
            "batter_name",
            "team",
            "opponent",
            "lineup_slot",
            "actual",
            "v3_prob",
            "savant_prob",
            "base_80_20",
            best_score,
            "v3_rank_daily",
            "savant_rank_daily",
            "prob_diff",
            "prob_abs_diff",
            "both_top10",
            "both_top25",
        ]
    ].rename(columns={best_score: "best_gated_score"})

    rank.to_csv(out_dir / "gated_blend_metrics.csv", index=False)
    top25.to_csv(out_dir / "best_gated_top25_daily_board.csv", index=False)
    df.to_parquet(out_dir / "all_gated_scores.parquet", index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "v3_preds": str(args.v3_preds),
        "savant_preds": str(args.savant_preds),
        "rows": int(len(df)),
        "dates": int(df["game_date"].nunique()),
        "best": best,
        "ranking": rank.to_dict(orient="records"),
        "outputs": {
            "metrics": str(out_dir / "gated_blend_metrics.csv"),
            "top25": str(out_dir / "best_gated_top25_daily_board.csv"),
            "all_scores": str(out_dir / "all_gated_scores.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
        "note": "Rule-based gated blend. No model training.",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps({
        "best": summary["best"],
        "top_results": summary["ranking"][:10],
        "outputs": summary["outputs"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
