#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Daily Blend Runner.

Production blend:
  80% Batter 1+ Hit Ranker v3
  20% Batter Savant-light challenger

Pipeline:
  1. Fetch confirmed MLB lineups
  2. Build v3 scoreable slate
  3. Add prior-season batter Savant features
  4. Score v3 model
  5. Score Savant-light model
  6. Blend probabilities
  7. Save daily top board
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date as dt_date
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SAFETY_VERSION = "batter_1plus_hit_daily_blend_v4_1_80v3_20savant_v3top5protect"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QEPC-MLB daily Batter 1+ Hit blended board")

    p.add_argument("--date", default=None, help="YYYY-MM-DD. Defaults to today.")
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--min_games", type=int, default=20)

    p.add_argument("--v3_model_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_champion_v3")
    p.add_argument("--savant_model_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_savant_light_challenger")

    p.add_argument("--v3_weight", type=float, default=0.80)
    p.add_argument("--savant_weight", type=float, default=0.20)

    p.add_argument("--allplayers", default="cache/mlb/raw/retrosheet/allplayers.csv")
    p.add_argument("--history", default="cache/mlb/processed/batter_game_logs_opp_pitcher_matchup.parquet")
    p.add_argument("--pitchers", default="cache/mlb/processed/pitcher_game_logs.parquet")
    p.add_argument("--batter_savant", default="cache/mlb/processed/batter_savant_features_2021_2025.parquet")

    p.add_argument("--manual_dir", default="cache/mlb/manual")
    p.add_argument("--scoring_dir", default="cache/mlb/scoring")
    p.add_argument("--prediction_dir", default="artifacts/mlb/predictions/batter_1plus_hit_blend")

    p.add_argument("--skip_fetch", action="store_true")
    p.add_argument("--lineup_csv", default=None)
    p.add_argument("--no_progress", action="store_true")

    return p.parse_args()


def run_cmd(cmd: List[str], step_name: str, progress=None) -> None:
    if progress is not None:
        progress.set_description(step_name)

    print("\n" + "=" * 100)
    print(f"RUNNING: {step_name}")
    print("=" * 100)
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        if progress is not None:
            progress.close()
        raise SystemExit(f"\nFAILED during step: {step_name}\nExit code: {result.returncode}")

    if progress is not None:
        progress.update(1)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def feature_lists(model_dir: Path) -> tuple[list[str], list[str]]:
    fl = load_json(model_dir / "feature_list.json")
    numeric = fl.get("numeric_features", [])
    categorical = fl.get("categorical_features", [])
    return list(numeric), list(categorical)


def prepare_x(df: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> pd.DataFrame:
    numeric_data = {}
    categorical_data = {}

    for c in numeric_features:
        if c in df.columns:
            numeric_data[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            numeric_data[c] = pd.Series(np.nan, index=df.index)

    for c in categorical_features:
        if c in df.columns:
            categorical_data[c] = df[c].astype("string").fillna("__MISSING__")
        else:
            categorical_data[c] = pd.Series("__MISSING__", index=df.index, dtype="string")

    x = pd.concat(
        [
            pd.DataFrame(numeric_data, index=df.index),
            pd.DataFrame(categorical_data, index=df.index),
        ],
        axis=1,
    )

    return x[list(numeric_features) + list(categorical_features)].copy()


def predict_model(df: pd.DataFrame, model_dir: Path) -> np.ndarray:
    model_path = model_dir / "model.joblib"
    model = joblib.load(model_path)
    numeric, categorical = feature_lists(model_dir)
    x = prepare_x(df, numeric, categorical)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def filter_scoreable(df: pd.DataFrame, min_games: int) -> pd.DataFrame:
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    starter_col = None
    for c in ["is_starting_lineup", "is_starter", "is_starting_lineup_by_batting_file"]:
        if c in out.columns:
            starter_col = c
            break

    if starter_col:
        starter = out[starter_col]
        if starter.dtype == bool:
            out = out[starter.fillna(False)].copy()
        else:
            out = out[pd.to_numeric(starter, errors="coerce").fillna(0).astype(float).ne(0)].copy()

    games_col = None
    for c in ["games_prior", "batter_games_prior"]:
        if c in out.columns:
            games_col = c
            break

    if games_col:
        out = out[pd.to_numeric(out[games_col], errors="coerce").fillna(0) >= min_games].copy()

    return out


def main() -> None:
    args = parse_args()

    if abs((args.v3_weight + args.savant_weight) - 1.0) > 1e-9:
        raise ValueError("--v3_weight + --savant_weight must equal 1.0")

    run_date = args.date or dt_date.today().isoformat()
    date_tag = run_date.replace("-", "")

    manual_dir = Path(args.manual_dir)
    scoring_dir = Path(args.scoring_dir)
    prediction_dir = Path(args.prediction_dir)

    manual_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    lineup_csv = Path(args.lineup_csv) if args.lineup_csv else manual_dir / f"mlb_confirmed_lineups_{date_tag}.csv"
    slate_v3 = scoring_dir / f"batter_1plus_hit_slate_{date_tag}.parquet"
    slate_savant = scoring_dir / f"batter_1plus_hit_slate_{date_tag}_batter_savant.parquet"

    fetch_script = "qepc_mlb/predict/fetch_mlb_confirmed_lineups.py"
    slate_script = "qepc_mlb/predict/build_batter_1plus_hit_scoring_slate.py"
    savant_join_script = "qepc_mlb/features/add_batter_prior_savant_quality.py"

    total_steps = 2 if args.skip_fetch else 3
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=total_steps, desc="QEPC blend runner", unit="step")

    if not args.skip_fetch:
        run_cmd(
            [
                sys.executable,
                fetch_script,
                "--date",
                run_date,
                "--allplayers",
                args.allplayers,
                "--out",
                str(lineup_csv),
            ],
            "Step 1/3: Fetch MLB confirmed lineups",
            progress=progress,
        )
    else:
        if not lineup_csv.exists():
            raise FileNotFoundError(f"--skip_fetch was passed, but lineup CSV does not exist: {lineup_csv}")
        print(f"Skipping fetch. Using existing lineup CSV: {lineup_csv}")

    run_cmd(
        [
            sys.executable,
            slate_script,
            "--lineup_csv",
            str(lineup_csv),
            "--history",
            args.history,
            "--pitchers",
            args.pitchers,
            "--model_dir",
            args.v3_model_dir,
            "--out",
            str(slate_v3),
            "--also_csv",
        ],
        "Step 2/3: Build v3 scoring slate",
        progress=progress,
    )

    run_cmd(
        [
            sys.executable,
            savant_join_script,
            "--batters",
            str(slate_v3),
            "--batter_savant",
            args.batter_savant,
            "--out",
            str(slate_savant),
        ],
        "Step 3/3: Add prior-season batter Savant features",
        progress=progress,
    )

    if progress is not None:
        progress.close()

    df = read_any(slate_savant)
    score_df = filter_scoreable(df, min_games=args.min_games).copy()

    if score_df.empty:
        raise SystemExit("No scoreable rows after filtering. Are confirmed lineups posted yet?")

    v3_model_dir = Path(args.v3_model_dir)
    sav_model_dir = Path(args.savant_model_dir)

    score_df["v3_prob"] = predict_model(score_df, v3_model_dir)
    score_df["savant_prob"] = predict_model(score_df, sav_model_dir)

    # Base production blend.
    score_df["blend_prob_base_80_20"] = (
        args.v3_weight * score_df["v3_prob"]
        + args.savant_weight * score_df["savant_prob"]
    )

    # QEPC v4.1 top-5 protection gate:
    # If v3 already ranks a player top-5 on the daily slate, keep the final score
    # slightly more v3-heavy. This improved top-5 hit rate while preserving top-10/top-25.
    score_df["v3_rank_daily"] = score_df.groupby("game_date")["v3_prob"].rank(
        method="first",
        ascending=False,
    )

    score_df["blend_prob"] = score_df["blend_prob_base_80_20"]
    top5_mask = score_df["v3_rank_daily"] <= 5
    score_df.loc[top5_mask, "blend_prob"] = (
        0.90 * score_df.loc[top5_mask, "v3_prob"]
        + 0.10 * score_df.loc[top5_mask, "savant_prob"]
    )

    score_df["pred_prob_pct"] = (score_df["blend_prob"] * 100).round(1)

    sort_cols = ["blend_prob"]
    ascending = [False]
    if "game_date" in score_df.columns:
        sort_cols = ["game_date", "blend_prob"]
        ascending = [True, False]

    top = (
        score_df.sort_values(sort_cols, ascending=ascending)
        .groupby("game_date", group_keys=False)
        .head(args.top_n)
        .copy()
        if "game_date" in score_df.columns
        else score_df.sort_values("blend_prob", ascending=False).head(args.top_n).copy()
    )

    top["pred_rank_daily"] = top.groupby("game_date")["blend_prob"].rank(method="first", ascending=False) if "game_date" in top.columns else range(1, len(top) + 1)

    display_cols = [
        "pred_rank_daily",
        "batter_name",
        "team",
        "opponent",
        "lineup_slot",
        "pred_prob_pct",
        "v3_prob",
        "savant_prob",
        "blend_prob_base_80_20",
        "v3_rank_daily",
        "blend_prob",
    ]

    for c in display_cols:
        if c not in top.columns:
            top[c] = pd.NA

    top = top[display_cols].copy()

    top_parquet = prediction_dir / f"batter_1plus_hit_blend_top{args.top_n}_{date_tag}.parquet"
    top_csv = prediction_dir / f"batter_1plus_hit_blend_top{args.top_n}_{date_tag}.csv"
    all_pred_parquet = prediction_dir / f"batter_1plus_hit_blend_all_predictions_{date_tag}.parquet"
    summary_path = prediction_dir / f"batter_1plus_hit_blend_summary_{date_tag}.json"

    top.to_parquet(top_parquet, index=False)
    top.to_csv(top_csv, index=False)
    score_df.to_parquet(all_pred_parquet, index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "date": run_date,
        "top_n": args.top_n,
        "v3_model_dir": args.v3_model_dir,
        "savant_model_dir": args.savant_model_dir,
        "v3_weight": args.v3_weight,
        "savant_weight": args.savant_weight,
        "gate": "v3_top5_protect",
        "gate_rule": "if v3_rank_daily <= 5, use 90% v3 + 10% Savant; otherwise use configured base blend",
        "lineup_csv": str(lineup_csv),
        "slate_v3": str(slate_v3),
        "slate_savant": str(slate_savant),
        "rows_raw": int(len(df)),
        "rows_scored": int(len(score_df)),
        "avg_v3_prob": float(score_df["v3_prob"].mean()),
        "avg_savant_prob": float(score_df["savant_prob"].mean()),
        "avg_blend_prob": float(score_df["blend_prob"].mean()),
        "top_board_avg_blend_prob": float(top["blend_prob"].mean()) if len(top) else None,
        "top_board_max_blend_prob": float(top["blend_prob"].max()) if len(top) else None,
        "outputs": {
            "top_board_parquet": str(top_parquet),
            "top_board_csv": str(top_csv),
            "all_predictions_parquet": str(all_pred_parquet),
            "summary": str(summary_path),
        },
        "note": "Production v4.1 gated blend: base 80% v3 + 20% Savant-light; v3 top-5 uses 90% v3 + 10% Savant.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100)
    print("DAILY BLEND PIPELINE COMPLETE")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))

    print("\nTop board preview:")
    print(top.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
