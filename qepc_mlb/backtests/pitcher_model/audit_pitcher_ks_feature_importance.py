#!/usr/bin/env python3
"""
QEPC-MLB Pitcher Strikeouts Champion Feature Importance Audit

Purpose
-------
Reads the trained champion model artifacts and summarizes what the model is
leaning on. This is an audit script, not a trainer.

Expected artifact directory:
    artifacts/mlb/models/pitcher_strikeouts_champion/

Expected files when available:
    model.joblib
    metadata.json
    feature_list.json
    feature_importances.csv
    training_predictions.parquet

Outputs:
    feature_importance_ranked.csv
    group_importance.csv
    top_features.json
    audit_summary.json
    optional residual diagnostics if training_predictions exists

Safety
------
Flags known leakage-prone lineup columns if they appear anywhere in the feature
importance table. These should NOT be present in a safe champion model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SAFETY_VERSION = "pitcher_ks_feature_importance_audit_v1"
LEAKY_FEATURES = {"lineup_batting_ks", "lineup_batting_pa"}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def normalize_feature_name(name: str) -> str:
    """Remove common sklearn pipeline prefixes and one-hot suffix clutter."""
    s = str(name)
    # sklearn ColumnTransformer prefixes, e.g. num__days_rest, cat__team_LAD
    s = re.sub(r"^(num|numeric|cat|categorical|remainder)__", "", s)
    return s


def base_feature_name(name: str) -> str:
    """
    Collapse one-hot encoded categorical features back to their source feature.

    Examples:
      team_LAD -> team
      opponent_NYY -> opponent
      env_park_id_LAD01 -> env_park_id
      pitcher_hand_R -> pitcher_hand
    """
    s = normalize_feature_name(name)

    known_cats = [
        "pitcher_hand",
        "team",
        "opponent",
        "home_away",
        "env_park_id",
        "env_winddir",
        "env_daynight",
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
        "has_lineup_context",
    ]
    for prefix in sorted(known_cats, key=len, reverse=True):
        if s == prefix or s.startswith(prefix + "_"):
            return prefix
    return s


def feature_group(name: str) -> str:
    b = base_feature_name(name)
    if b in LEAKY_FEATURES:
        return "LEAKY_DO_NOT_USE"
    if b.startswith("savant_"):
        return "savant_skill"
    if b.startswith("opp_team_"):
        return "opponent_context"
    if b.startswith("lineup_") or b == "has_lineup_context":
        return "lineup_context_safe"
    if b.startswith("env_") or b == "has_env_context":
        return "environment_context"
    if b in {"team", "opponent", "home_away", "pitcher_hand"}:
        return "categorical_identity_context"
    if b in {
        "batters_faced",
        "innings_pitched",
        "outs_recorded",
        "days_rest",
        "walks",
        "hits_allowed",
        "earned_runs",
        "runs_allowed",
        "games_prior",
    }:
        return "current_game_or_pitcher_state"
    if (
        b.startswith("k_roll_")
        or b.startswith("target_lag")
        or b.endswith("_lag1")
        or b.endswith("_roll5")
        or b.startswith("k_per_bf")
    ):
        return "pitcher_recent_form"
    return "other"


def detect_importance_columns(df: pd.DataFrame) -> Tuple[str, str]:
    feature_candidates = ["feature", "feature_name", "name", "features"]
    importance_candidates = ["importance", "feature_importance", "gain", "weight"]

    feature_col = next((c for c in feature_candidates if c in df.columns), None)
    importance_col = next((c for c in importance_candidates if c in df.columns), None)

    if feature_col is None:
        # Common fallback: first object-ish column.
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if object_cols:
            feature_col = object_cols[0]

    if importance_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            # Avoid rank/index columns if possible.
            non_rank = [c for c in numeric_cols if c.lower() not in {"rank", "index", "level_0"}]
            importance_col = non_rank[0] if non_rank else numeric_cols[0]

    if feature_col is None or importance_col is None:
        raise ValueError(
            "Could not detect feature/importance columns. "
            f"Available columns: {list(df.columns)}"
        )
    return feature_col, importance_col


def load_feature_importances(artifact_dir: Path) -> pd.DataFrame:
    imp_path = artifact_dir / "feature_importances.csv"
    if not imp_path.exists():
        raise FileNotFoundError(
            f"Missing {imp_path}. Re-run train_pitcher_ks_champion.py to create feature_importances.csv."
        )

    raw = pd.read_csv(imp_path)
    feature_col, importance_col = detect_importance_columns(raw)

    out = raw[[feature_col, importance_col]].copy()
    out.columns = ["feature", "importance"]
    out["feature"] = out["feature"].astype(str)
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0)

    out["clean_feature"] = out["feature"].map(normalize_feature_name)
    out["base_feature"] = out["feature"].map(base_feature_name)
    out["feature_group"] = out["feature"].map(feature_group)

    total = float(out["importance"].sum())
    if total > 0:
        out["importance_share"] = out["importance"] / total
    else:
        out["importance_share"] = 0.0

    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def summarize_groups(importance_df: pd.DataFrame) -> pd.DataFrame:
    group = (
        importance_df.groupby("feature_group", dropna=False)
        .agg(
            total_importance=("importance", "sum"),
            total_importance_share=("importance_share", "sum"),
            encoded_feature_count=("feature", "count"),
            base_feature_count=("base_feature", "nunique"),
            top_feature=("base_feature", lambda s: s.iloc[0] if len(s) else None),
        )
        .reset_index()
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )
    group.insert(0, "rank", np.arange(1, len(group) + 1))
    return group


def summarize_base_features(importance_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        importance_df.groupby(["base_feature", "feature_group"], dropna=False)
        .agg(
            total_importance=("importance", "sum"),
            total_importance_share=("importance_share", "sum"),
            encoded_feature_count=("feature", "count"),
        )
        .reset_index()
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )
    base.insert(0, "rank", np.arange(1, len(base) + 1))
    return base


def load_predictions(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read predictions file {path}: {exc}")
        return None


def find_col(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def residual_diagnostics(preds: pd.DataFrame) -> Dict[str, Any]:
    actual_col = find_col(preds.columns, ["actual", "actual_strikeouts", "strikeouts", "y_true", "target"])
    pred_col = find_col(preds.columns, ["prediction", "predicted", "pred", "predicted_strikeouts", "y_pred"])

    if actual_col is None or pred_col is None:
        return {
            "available": False,
            "reason": "Could not detect actual/prediction columns.",
            "columns": list(preds.columns),
        }

    d = preds[[actual_col, pred_col]].copy()
    d[actual_col] = pd.to_numeric(d[actual_col], errors="coerce")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return {"available": False, "reason": "No numeric actual/prediction rows after cleaning."}

    err = d[pred_col] - d[actual_col]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # Bucket by prediction level to spot systematic miss patterns.
    try:
        bucket = pd.cut(d[pred_col], bins=[-np.inf, 3, 5, 7, np.inf], labels=["<=3", "3-5", "5-7", ">7"])
        bucket_df = (
            pd.DataFrame({"bucket": bucket, "err": err, "abs_err": np.abs(err)})
            .groupby("bucket", observed=False)
            .agg(rows=("err", "size"), mae=("abs_err", "mean"), bias=("err", "mean"))
            .reset_index()
        )
        bucket_records = bucket_df.to_dict(orient="records")
    except Exception:
        bucket_records = []

    return {
        "available": True,
        "actual_col": actual_col,
        "prediction_col": pred_col,
        "rows": int(len(d)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "prediction_bucket_diagnostics": bucket_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit QEPC-MLB pitcher strikeout champion feature importances.")
    parser.add_argument(
        "--artifact_dir",
        default="artifacts/mlb/models/pitcher_strikeouts_champion",
        help="Directory containing champion model artifacts.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/mlb/audits/pitcher_ks_feature_importance",
        help="Directory to write audit outputs.",
    )
    parser.add_argument("--top_n", type=int, default=40, help="Number of top features to include in JSON summary.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional predictions file for residual diagnostics. Defaults to training_predictions.parquet in artifact_dir.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(artifact_dir / "metadata.json", default={})
    feature_list = read_json(artifact_dir / "feature_list.json", default={})

    imp = load_feature_importances(artifact_dir)
    group = summarize_groups(imp)
    base = summarize_base_features(imp)

    leaky_found = sorted(set(imp[imp["base_feature"].isin(LEAKY_FEATURES)]["base_feature"].tolist()))

    imp_path = out_dir / "feature_importance_ranked.csv"
    group_path = out_dir / "group_importance.csv"
    base_path = out_dir / "base_feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    group.to_csv(group_path, index=False)
    base.to_csv(base_path, index=False)

    pred_path = Path(args.predictions) if args.predictions else artifact_dir / "training_predictions.parquet"
    preds = load_predictions(pred_path)
    residuals = residual_diagnostics(preds) if preds is not None else {"available": False, "reason": "No predictions file found."}

    top_features = base.head(args.top_n).to_dict(orient="records")
    write_json(out_dir / "top_features.json", top_features)

    summary: Dict[str, Any] = {
        "safety_version": SAFETY_VERSION,
        "artifact_dir": str(artifact_dir),
        "out_dir": str(out_dir),
        "metadata_safety_version": metadata.get("safety_version"),
        "champion_config": metadata.get("champion_config"),
        "rows_after_filtering": metadata.get("rows_after_filtering"),
        "seasons": metadata.get("seasons"),
        "pitchers": metadata.get("pitchers"),
        "encoded_feature_count": int(len(imp)),
        "base_feature_count": int(base["base_feature"].nunique()),
        "feature_group_count": int(group["feature_group"].nunique()),
        "leaky_features_found_in_importances": leaky_found,
        "top_groups": group.head(15).to_dict(orient="records"),
        "top_features": top_features,
        "residual_diagnostics": residuals,
        "outputs": {
            "feature_importance_ranked": str(imp_path),
            "group_importance": str(group_path),
            "base_feature_importance": str(base_path),
            "top_features": str(out_dir / "top_features.json"),
            "audit_summary": str(out_dir / "audit_summary.json"),
        },
    }

    write_json(out_dir / "audit_summary.json", summary)

    print(f"saved feature importances: {imp_path}")
    print(f"saved group importance: {group_path}")
    print(f"saved base feature importance: {base_path}")
    print(f"saved summary: {out_dir / 'audit_summary.json'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
