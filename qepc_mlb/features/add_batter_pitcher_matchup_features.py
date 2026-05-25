#!/usr/bin/env python3
"""
Add batter-vs-opposing-starter handedness matchup features.

Input:
  cache/mlb/processed/batter_game_logs_opp_pitcher.parquet

Output:
  cache/mlb/processed/batter_game_logs_opp_pitcher_matchup.parquet

Safety:
  These are pregame-safe identity/context features only.
  No same-game outcomes are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAFETY_VERSION = "batter_pitcher_matchup_features_v1_pregame_safe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--also_csv", action="store_true")
    return p.parse_args()


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_hand(x: Any) -> str:
    if pd.isna(x):
        return "U"

    s = str(x).strip().upper()

    if s in {"R", "RIGHT", "RIGHTY"}:
        return "R"
    if s in {"L", "LEFT", "LEFTY"}:
        return "L"
    if s in {"S", "B", "SW", "SWITCH", "BOTH"}:
        return "S"

    return "U"


def main() -> None:
    args = parse_args()

    df = read_any(args.input).copy()

    if "batter_hand" not in df.columns:
        raise ValueError("Missing batter_hand column.")
    if "opp_starter_throw" not in df.columns:
        raise ValueError("Missing opp_starter_throw column.")

    df["batter_hand_norm"] = df["batter_hand"].map(normalize_hand)
    df["opp_starter_throw_norm"] = df["opp_starter_throw"].map(normalize_hand)

    df["batter_bats_right"] = df["batter_hand_norm"].eq("R").astype(int)
    df["batter_bats_left"] = df["batter_hand_norm"].eq("L").astype(int)
    df["batter_bats_switch"] = df["batter_hand_norm"].eq("S").astype(int)

    df["opp_starter_throws_right"] = df["opp_starter_throw_norm"].eq("R").astype(int)
    df["opp_starter_throws_left"] = df["opp_starter_throw_norm"].eq("L").astype(int)

    known_batter = df["batter_hand_norm"].isin(["R", "L", "S"])
    known_pitcher = df["opp_starter_throw_norm"].isin(["R", "L"])
    known_matchup = known_batter & known_pitcher

    same_hand = (
        (df["batter_hand_norm"].eq("R") & df["opp_starter_throw_norm"].eq("R"))
        | (df["batter_hand_norm"].eq("L") & df["opp_starter_throw_norm"].eq("L"))
    )

    opposite_hand = (
        (df["batter_hand_norm"].eq("R") & df["opp_starter_throw_norm"].eq("L"))
        | (df["batter_hand_norm"].eq("L") & df["opp_starter_throw_norm"].eq("R"))
    )

    # Switch hitters are treated as having a platoon advantage when pitcher hand is known.
    platoon_advantage = opposite_hand | (df["batter_hand_norm"].eq("S") & known_pitcher)

    df["matchup_known"] = known_matchup.astype(int)
    df["same_hand_matchup"] = np.where(known_matchup, same_hand.astype(int), np.nan)
    df["opposite_hand_matchup"] = np.where(known_matchup, opposite_hand.astype(int), np.nan)
    df["platoon_advantage"] = np.where(known_matchup, platoon_advantage.astype(int), np.nan)

    df["batter_pitcher_matchup_code"] = (
        df["batter_hand_norm"].astype(str)
        + "_vs_"
        + df["opp_starter_throw_norm"].astype(str)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    new_cols = [
        "batter_hand_norm",
        "opp_starter_throw_norm",
        "batter_bats_right",
        "batter_bats_left",
        "batter_bats_switch",
        "opp_starter_throws_right",
        "opp_starter_throws_left",
        "matchup_known",
        "same_hand_matchup",
        "opposite_hand_matchup",
        "platoon_advantage",
        "batter_pitcher_matchup_code",
    ]

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": args.input,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "rows": int(len(df)),
        "new_columns": new_cols,
        "matchup_known_rate": float(df["matchup_known"].mean()),
        "platoon_advantage_rate_known_rows": float(
            df.loc[df["matchup_known"].eq(1), "platoon_advantage"].mean()
        ) if df["matchup_known"].sum() else None,
        "batter_hand_counts": df["batter_hand_norm"].value_counts(dropna=False).to_dict(),
        "opp_starter_throw_counts": df["opp_starter_throw_norm"].value_counts(dropna=False).to_dict(),
        "note": "Pregame-safe handedness matchup features only.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
