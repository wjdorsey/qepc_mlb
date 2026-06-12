#!/usr/bin/env python3
"""
Diagnose where S2 Statcast improvements come from.

Crosses out-of-fold S1 and S2 component predictions:

- S1 BF × S1 K-rate
- S2 BF × S1 K-rate
- S1 BF × S2 K-rate
- S2 BF × S2 K-rate
- 50/50 final-projection blend
- 50/50 component blend

No model is retrained. Every component prediction is already
walk-forward/out-of-fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from walk_forward_pitcher_ks_s0 import (
    point_metrics,
    ranking_metrics,
)


SAFETY_VERSION = "pitcher_ks_s2_component_routing_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_pregame_s1_decomposed/"
            "predictions.parquet"
        ),
    )

    parser.add_argument(
        "--s2",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_pregame_s2_statcast/"
            "predictions.parquet"
        ),
    )

    parser.add_argument(
        "--source_model",
        default="decomp_bf_poisson_rate_logit",
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_s2_component_routing"
        ),
    )

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input type: {path}")


def choose_keys(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
) -> list[str]:
    preferred = ["game_id", "pitcher_id"]

    if all(
        column in s1.columns and column in s2.columns
        for column in preferred
    ):
        return preferred

    fallback = [
        "game_date",
        "pitcher_id",
        "team",
        "opponent",
    ]

    keys = [
        column
        for column in fallback
        if column in s1.columns
        and column in s2.columns
    ]

    if not {
        "game_date",
        "pitcher_id",
    }.issubset(keys):
        raise ValueError(
            "Could not construct reliable paired keys."
        )

    return keys


def summarize(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    point = point_metrics(
        frame["actual_strikeouts"],
        frame["predicted_strikeouts"],
    )

    ranking = ranking_metrics(frame)

    return {
        **point,
        **ranking,
        "rows_scored": int(len(frame)),
        "dates": int(
            frame["game_date"].nunique()
        ),
    }


def main() -> None:
    args = parse_args()

    s1_path = Path(args.s1)
    s2_path = Path(args.s2)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = read_table(s1_path)
    s2 = read_table(s2_path)

    s1 = s1[
        s1["model"].eq(args.source_model)
    ].copy()

    s2 = s2[
        s2["model"].eq(args.source_model)
    ].copy()

    if s1.empty or s2.empty:
        raise ValueError(
            "Requested source model was missing from "
            "one or both prediction files."
        )

    keys = choose_keys(s1, s2)

    required_components = [
        "predicted_strikeouts",
        "predicted_batters_faced",
        "predicted_k_per_bf",
        "actual_strikeouts",
        "game_date",
    ]

    for label, frame in [
        ("S1", s1),
        ("S2", s2),
    ]:
        missing = [
            column
            for column in required_components
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{label} missing columns: {missing}"
            )

        duplicates = frame.duplicated(
            keys,
            keep=False,
        )

        if duplicates.any():
            raise RuntimeError(
                f"{label} has duplicate paired keys."
            )

    metadata = [
        column
        for column in [
            "game_date",
            "season",
            "pitcher_name",
            "team",
            "opponent",
            "period",
        ]
        if column in s1.columns
        and column not in keys
    ]

    s1_keep = (
        keys
        + metadata
        + [
            "actual_strikeouts",
            "predicted_strikeouts",
            "predicted_batters_faced",
            "predicted_k_per_bf",
        ]
    )

    left = s1[s1_keep].rename(
        columns={
            "predicted_strikeouts": "s1_predicted_k",
            "predicted_batters_faced": "s1_predicted_bf",
            "predicted_k_per_bf": "s1_predicted_rate",
        }
    )

    right = s2[
        keys
        + [
            "predicted_strikeouts",
            "predicted_batters_faced",
            "predicted_k_per_bf",
        ]
    ].rename(
        columns={
            "predicted_strikeouts": "s2_predicted_k",
            "predicted_batters_faced": "s2_predicted_bf",
            "predicted_k_per_bf": "s2_predicted_rate",
        }
    )

    paired = left.merge(
        right,
        how="inner",
        on=keys,
        validate="one_to_one",
    )

    needed = [
        "actual_strikeouts",
        "s1_predicted_k",
        "s1_predicted_bf",
        "s1_predicted_rate",
        "s2_predicted_k",
        "s2_predicted_bf",
        "s2_predicted_rate",
    ]

    paired = paired.dropna(
        subset=needed
    ).copy()

    if paired.empty:
        raise RuntimeError(
            "No complete paired rows remained."
        )

    average_bf = (
        paired["s1_predicted_bf"]
        + paired["s2_predicted_bf"]
    ) / 2.0

    average_rate = (
        paired["s1_predicted_rate"]
        + paired["s2_predicted_rate"]
    ) / 2.0

    candidates = {
        "s1_incumbent": (
            paired["s1_predicted_k"]
        ),
        "routed_s2_bf_s1_rate": (
            paired["s2_predicted_bf"]
            * paired["s1_predicted_rate"]
        ),
        "routed_s1_bf_s2_rate": (
            paired["s1_predicted_bf"]
            * paired["s2_predicted_rate"]
        ),
        "s2_full": (
            paired["s2_predicted_k"]
        ),
        "blend_final_50_50": (
            (
                paired["s1_predicted_k"]
                + paired["s2_predicted_k"]
            )
            / 2.0
        ),
        "blend_components_50_50": (
            average_bf * average_rate
        ),
    }

    prediction_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    base_columns = (
        keys
        + metadata
        + ["actual_strikeouts"]
    )

    for model_name, prediction in candidates.items():
        frame = paired[
            base_columns
        ].copy()

        frame["model"] = model_name
        frame["predicted_strikeouts"] = np.clip(
            pd.to_numeric(
                prediction,
                errors="coerce",
            ),
            0.0,
            None,
        )

        frame = frame.dropna(
            subset=[
                "actual_strikeouts",
                "predicted_strikeouts",
            ]
        )

        prediction_frames.append(frame)

        summary = summarize(frame)
        summary["model"] = model_name
        summaries.append(summary)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    ranking_by_mae = sorted(
        summaries,
        key=lambda row: (
            row["mae"],
            row["rmse"],
        ),
    )

    predictions_path = (
        out_dir / "predictions.parquet"
    )

    summary_path = (
        out_dir / "summary.json"
    )

    comparison_path = (
        out_dir / "comparison.csv"
    )

    all_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    comparison = pd.DataFrame(
        ranking_by_mae
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    final = {
        "safety_version": SAFETY_VERSION,
        "s1_predictions": str(s1_path),
        "s2_predictions": str(s2_path),
        "source_model": args.source_model,
        "paired_keys": keys,
        "paired_rows": int(len(paired)),
        "paired_dates": int(
            paired["game_date"].nunique()
        ),
        "models_tested": list(candidates),
        "ranking_by_mae": ranking_by_mae,
        "best_model": ranking_by_mae[0],
        "outputs": {
            "predictions": str(predictions_path),
            "comparison": str(comparison_path),
            "summary": str(summary_path),
        },
        "interpretation": {
            "routed_s2_bf_s1_rate": (
                "Tests whether Statcast improves workload only."
            ),
            "routed_s1_bf_s2_rate": (
                "Tests whether Statcast improves K-rate only."
            ),
            "blend_final_50_50": (
                "Fixed average of S1 and S2 total-K projections."
            ),
            "blend_components_50_50": (
                "Fixed average BF multiplied by fixed average K-rate."
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            final,
            indent=2,
            default=str,
        )
    )

    print(
        json.dumps(
            final,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
