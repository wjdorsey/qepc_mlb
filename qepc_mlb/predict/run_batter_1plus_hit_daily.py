#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Daily Runner.

One-command pipeline:

  1. Fetch MLB confirmed lineups from MLB Stats API
  2. Build prediction-ready scoring slate
  3. Score Batter 1+ Hit Ranker v2
  4. Save top board CSV/parquet

Example:

  python qepc_mlb/predict/run_batter_1plus_hit_daily.py \
    --date 2025-09-28 \
    --top_n 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date as dt_date
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SAFETY_VERSION = "batter_1plus_hit_daily_runner_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QEPC-MLB Batter 1+ Hit daily board pipeline")

    p.add_argument("--date", default=None, help="YYYY-MM-DD. Defaults to today.")
    p.add_argument("--top_n", type=int, default=25)

    p.add_argument("--allplayers", default="cache/mlb/raw/retrosheet/allplayers.csv")
    p.add_argument("--history", default="cache/mlb/processed/batter_game_logs_opp_pitcher.parquet")
    p.add_argument("--pitchers", default="cache/mlb/processed/pitcher_game_logs.parquet")
    p.add_argument("--model_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_champion_v2")

    p.add_argument("--manual_dir", default="cache/mlb/manual")
    p.add_argument("--scoring_dir", default="cache/mlb/scoring")
    p.add_argument("--prediction_dir", default="artifacts/mlb/predictions/batter_1plus_hit_ranker")

    p.add_argument("--include_non_starters", action="store_true")
    p.add_argument("--min_games", type=int, default=20)
    p.add_argument("--save_all_predictions", action="store_true")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar")

    p.add_argument("--skip_fetch", action="store_true", help="Use an existing lineup CSV instead of fetching")
    p.add_argument("--lineup_csv", default=None, help="Required if --skip_fetch is used")

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


def main() -> None:
    args = parse_args()

    run_date = args.date or dt_date.today().isoformat()
    date_tag = run_date.replace("-", "")

    manual_dir = Path(args.manual_dir)
    scoring_dir = Path(args.scoring_dir)
    prediction_dir = Path(args.prediction_dir)

    manual_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    lineup_csv = Path(args.lineup_csv) if args.lineup_csv else manual_dir / f"mlb_confirmed_lineups_{date_tag}.csv"
    slate_path = scoring_dir / f"batter_1plus_hit_slate_{date_tag}.parquet"

    fetch_script = "qepc_mlb/predict/fetch_mlb_confirmed_lineups.py"
    slate_script = "qepc_mlb/predict/build_batter_1plus_hit_scoring_slate.py"
    predict_script = "qepc_mlb/predict/predict_batter_1plus_hit_ranker.py"

    total_steps = 2 if args.skip_fetch else 3
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=total_steps, desc="QEPC daily runner", unit="step")

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
            args.model_dir,
            "--out",
            str(slate_path),
            "--also_csv",
        ],
        "Step 2/3: Build scoring slate",
        progress=progress,
    )

    predict_cmd = [
        sys.executable,
        predict_script,
        "--input",
        str(slate_path),
        "--model_dir",
        args.model_dir,
        "--date",
        run_date,
        "--top_n",
        str(args.top_n),
        "--min_games",
        str(args.min_games),
        "--also_csv",
    ]

    if args.include_non_starters:
        predict_cmd.append("--include_non_starters")

    if args.save_all_predictions:
        predict_cmd.append("--save_all_predictions")

    run_cmd(
        predict_cmd,
        "Step 3/3: Score Batter 1+ Hit board",
        progress=progress,
    )

    top_board_csv = prediction_dir / f"batter_1plus_hit_top{args.top_n}_{date_tag}.csv"
    top_board_parquet = prediction_dir / f"batter_1plus_hit_top{args.top_n}_{date_tag}.parquet"
    pred_summary = prediction_dir / f"batter_1plus_hit_prediction_summary_{date_tag}.json"

    summary: Dict[str, Any] = {
        "safety_version": SAFETY_VERSION,
        "date": run_date,
        "top_n": args.top_n,
        "lineup_csv": str(lineup_csv),
        "slate_parquet": str(slate_path),
        "slate_csv": str(slate_path.with_suffix(".csv")),
        "top_board_csv": str(top_board_csv),
        "top_board_parquet": str(top_board_parquet),
        "prediction_summary": str(pred_summary),
        "model_dir": args.model_dir,
        "note": "Daily Batter 1+ Hit pipeline completed.",
    }

    runner_summary = prediction_dir / f"batter_1plus_hit_daily_runner_summary_{date_tag}.json"
    runner_summary.write_text(json.dumps(summary, indent=2, default=str))

    if progress is not None:
        progress.close()

    print("\n" + "=" * 100)
    print("DAILY PIPELINE COMPLETE")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
