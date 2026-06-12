#!/usr/bin/env python3
"""
Paired date-block comparison for QEPC-MLB pitcher strikeout models.

The bootstrap samples complete game dates rather than individual pitcher rows.
This preserves same-slate correlation.

Default comparison:
    direct_hgb_squared
vs.
    decomp_bf_poisson_rate_logit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score


SAFETY_VERSION = "pitcher_ks_paired_date_bootstrap_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_pregame_s1_decomposed/"
            "predictions.parquet"
        ),
    )

    parser.add_argument(
        "--control",
        default="direct_hgb_squared",
    )

    parser.add_argument(
        "--challenger",
        default="decomp_bf_poisson_rate_logit",
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_pregame_s1_decomposed/"
            "paired_comparison"
        ),
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
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


def choose_keys(df: pd.DataFrame) -> list[str]:
    if {
        "game_id",
        "pitcher_id",
    }.issubset(df.columns):
        return ["game_id", "pitcher_id"]

    fallback = [
        "game_date",
        "pitcher_id",
        "team",
        "opponent",
    ]

    keys = [
        column
        for column in fallback
        if column in df.columns
    ]

    if not {
        "game_date",
        "pitcher_id",
    }.issubset(keys):
        raise ValueError(
            "Could not construct a reliable paired-row key."
        )

    return keys


def safe_ndcg(
    actual: np.ndarray,
    predicted: np.ndarray,
    k: int,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) < 2 or actual.sum() <= 0:
        return np.nan

    use_k = min(k, len(actual))

    return float(
        ndcg_score(
            actual.reshape(1, -1),
            predicted.reshape(1, -1),
            k=use_k,
        )
    )


def top_board_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    k: int,
) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    use_k = min(k, len(actual))

    if use_k == 0:
        return np.nan, np.nan

    predicted_order = np.argsort(-predicted)[:use_k]
    oracle_order = np.argsort(-actual)[:use_k]

    selected_actual_mean = float(
        actual[predicted_order].mean()
    )

    regret = float(
        actual[oracle_order].sum()
        - actual[predicted_order].sum()
    )

    return selected_actual_mean, regret


def model_summary(
    paired: pd.DataFrame,
    prefix: str,
) -> dict[str, float]:
    actual = paired["actual_strikeouts"].to_numpy(
        dtype=float
    )

    predicted = paired[
        f"{prefix}_prediction"
    ].to_numpy(dtype=float)

    residual = predicted - actual

    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(
            math.sqrt(np.mean(residual ** 2))
        ),
        "bias": float(np.mean(residual)),
        "median_bias": float(np.median(residual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_mean": float(np.mean(actual)),
    }


def build_daily_metrics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for game_date, day in paired.groupby(
        "game_date",
        sort=True,
    ):
        actual = day["actual_strikeouts"].to_numpy(
            dtype=float
        )

        control = day[
            "control_prediction"
        ].to_numpy(dtype=float)

        challenger = day[
            "challenger_prediction"
        ].to_numpy(dtype=float)

        control_residual = control - actual
        challenger_residual = challenger - actual

        control_top5, control_regret5 = (
            top_board_metrics(
                actual,
                control,
                k=5,
            )
        )

        challenger_top5, challenger_regret5 = (
            top_board_metrics(
                actual,
                challenger,
                k=5,
            )
        )

        control_top10, control_regret10 = (
            top_board_metrics(
                actual,
                control,
                k=10,
            )
        )

        challenger_top10, challenger_regret10 = (
            top_board_metrics(
                actual,
                challenger,
                k=10,
            )
        )

        row = {
            "game_date": game_date,
            "rows": int(len(day)),
            "control_abs_error_sum": float(
                np.abs(control_residual).sum()
            ),
            "challenger_abs_error_sum": float(
                np.abs(challenger_residual).sum()
            ),
            "control_sq_error_sum": float(
                np.square(control_residual).sum()
            ),
            "challenger_sq_error_sum": float(
                np.square(challenger_residual).sum()
            ),
            "control_residual_sum": float(
                control_residual.sum()
            ),
            "challenger_residual_sum": float(
                challenger_residual.sum()
            ),
            "control_mae": float(
                np.mean(np.abs(control_residual))
            ),
            "challenger_mae": float(
                np.mean(np.abs(challenger_residual))
            ),
            "control_rmse": float(
                math.sqrt(
                    np.mean(
                        np.square(control_residual)
                    )
                )
            ),
            "challenger_rmse": float(
                math.sqrt(
                    np.mean(
                        np.square(challenger_residual)
                    )
                )
            ),
            "control_bias": float(
                np.mean(control_residual)
            ),
            "challenger_bias": float(
                np.mean(challenger_residual)
            ),
            "control_ndcg5": safe_ndcg(
                actual,
                control,
                k=5,
            ),
            "challenger_ndcg5": safe_ndcg(
                actual,
                challenger,
                k=5,
            ),
            "control_ndcg10": safe_ndcg(
                actual,
                control,
                k=10,
            ),
            "challenger_ndcg10": safe_ndcg(
                actual,
                challenger,
                k=10,
            ),
            "control_top5_actual_k_mean": (
                control_top5
            ),
            "challenger_top5_actual_k_mean": (
                challenger_top5
            ),
            "control_top10_actual_k_mean": (
                control_top10
            ),
            "challenger_top10_actual_k_mean": (
                challenger_top10
            ),
            "control_top5_regret": (
                control_regret5
            ),
            "challenger_top5_regret": (
                challenger_regret5
            ),
            "control_top10_regret": (
                control_regret10
            ),
            "challenger_top10_regret": (
                challenger_regret10
            ),
        }

        row[
            "mae_delta_challenger_minus_control"
        ] = (
            row["challenger_mae"]
            - row["control_mae"]
        )

        row[
            "rmse_delta_challenger_minus_control"
        ] = (
            row["challenger_rmse"]
            - row["control_rmse"]
        )

        row[
            "ndcg5_delta_challenger_minus_control"
        ] = (
            row["challenger_ndcg5"]
            - row["control_ndcg5"]
        )

        row[
            "top5_regret_delta_challenger_minus_control"
        ] = (
            row["challenger_top5_regret"]
            - row["control_top5_regret"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_sample(
    sampled: pd.DataFrame,
) -> dict[str, float]:
    total_rows = sampled["rows"].sum()

    control_mae = (
        sampled["control_abs_error_sum"].sum()
        / total_rows
    )

    challenger_mae = (
        sampled["challenger_abs_error_sum"].sum()
        / total_rows
    )

    control_rmse = math.sqrt(
        sampled["control_sq_error_sum"].sum()
        / total_rows
    )

    challenger_rmse = math.sqrt(
        sampled["challenger_sq_error_sum"].sum()
        / total_rows
    )

    control_bias = (
        sampled["control_residual_sum"].sum()
        / total_rows
    )

    challenger_bias = (
        sampled["challenger_residual_sum"].sum()
        / total_rows
    )

    return {
        "mae_delta": (
            challenger_mae - control_mae
        ),
        "rmse_delta": (
            challenger_rmse - control_rmse
        ),
        "bias_abs_delta": (
            abs(challenger_bias)
            - abs(control_bias)
        ),
        "ndcg5_delta": float(
            sampled[
                "challenger_ndcg5"
            ].mean()
            - sampled[
                "control_ndcg5"
            ].mean()
        ),
        "ndcg10_delta": float(
            sampled[
                "challenger_ndcg10"
            ].mean()
            - sampled[
                "control_ndcg10"
            ].mean()
        ),
        "top5_actual_k_delta": float(
            sampled[
                "challenger_top5_actual_k_mean"
            ].mean()
            - sampled[
                "control_top5_actual_k_mean"
            ].mean()
        ),
        "top5_regret_delta": float(
            sampled[
                "challenger_top5_regret"
            ].mean()
            - sampled[
                "control_top5_regret"
            ].mean()
        ),
    }


def interval_summary(
    values: np.ndarray,
    better_direction: str,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)

    if better_direction == "lower":
        probability_better = float(
            np.mean(values < 0)
        )
    elif better_direction == "higher":
        probability_better = float(
            np.mean(values > 0)
        )
    else:
        raise ValueError(better_direction)

    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_2_5": float(
            np.quantile(values, 0.025)
        ),
        "ci_97_5": float(
            np.quantile(values, 0.975)
        ),
        "probability_challenger_better": (
            probability_better
        ),
    }


def build_monthly_metrics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    frame = paired.copy()

    frame["month"] = (
        pd.to_datetime(frame["game_date"])
        .dt.to_period("M")
        .astype(str)
    )

    rows: list[dict[str, Any]] = []

    for month, group in frame.groupby(
        "month",
        sort=True,
    ):
        control = model_summary(
            group,
            prefix="control",
        )

        challenger = model_summary(
            group,
            prefix="challenger",
        )

        rows.append(
            {
                "month": month,
                "rows": int(len(group)),
                "control_mae": control["mae"],
                "challenger_mae": challenger["mae"],
                "mae_delta_challenger_minus_control": (
                    challenger["mae"]
                    - control["mae"]
                ),
                "control_rmse": control["rmse"],
                "challenger_rmse": challenger["rmse"],
                "rmse_delta_challenger_minus_control": (
                    challenger["rmse"]
                    - control["rmse"]
                ),
                "control_bias": control["bias"],
                "challenger_bias": challenger["bias"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_table(predictions_path)

    required = {
        "model",
        "game_date",
        "actual_strikeouts",
        "predicted_strikeouts",
    }

    missing = sorted(
        required - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    predictions["game_date"] = pd.to_datetime(
        predictions["game_date"],
        errors="coerce",
    )

    selected = predictions[
        predictions["model"].isin(
            [args.control, args.challenger]
        )
    ].copy()

    present_models = set(
        selected["model"].unique()
    )

    expected_models = {
        args.control,
        args.challenger,
    }

    if present_models != expected_models:
        raise ValueError(
            "Requested models not both present. "
            f"Found: {sorted(present_models)}"
        )

    keys = choose_keys(selected)

    metadata_columns = [
        column
        for column in [
            "game_date",
            "season",
            "pitcher_name",
            "team",
            "opponent",
        ]
        if column in selected.columns
        and column not in keys
    ]

    duplicate_rows = selected.duplicated(
        keys + ["model"],
        keep=False,
    )

    if duplicate_rows.any():
        raise RuntimeError(
            "Duplicate model predictions found for paired keys."
        )

    control = selected[
        selected["model"].eq(args.control)
    ].copy()

    challenger = selected[
        selected["model"].eq(args.challenger)
    ].copy()

    control_keep = (
        keys
        + metadata_columns
        + [
            "actual_strikeouts",
            "predicted_strikeouts",
        ]
    )

    control = control[control_keep].rename(
        columns={
            "predicted_strikeouts": (
                "control_prediction"
            )
        }
    )

    challenger = challenger[
        keys + ["predicted_strikeouts"]
    ].rename(
        columns={
            "predicted_strikeouts": (
                "challenger_prediction"
            )
        }
    )

    paired = control.merge(
        challenger,
        how="inner",
        on=keys,
        validate="one_to_one",
    )

    paired = paired.dropna(
        subset=[
            "game_date",
            "actual_strikeouts",
            "control_prediction",
            "challenger_prediction",
        ]
    ).copy()

    if paired.empty:
        raise RuntimeError(
            "No paired predictions remained."
        )

    control_summary = model_summary(
        paired,
        prefix="control",
    )

    challenger_summary = model_summary(
        paired,
        prefix="challenger",
    )

    daily = build_daily_metrics(paired)
    monthly = build_monthly_metrics(paired)

    rng = np.random.default_rng(
        args.random_state
    )

    bootstrap_results: dict[str, list[float]] = {
        "mae_delta": [],
        "rmse_delta": [],
        "bias_abs_delta": [],
        "ndcg5_delta": [],
        "ndcg10_delta": [],
        "top5_actual_k_delta": [],
        "top5_regret_delta": [],
    }

    daily_indices = np.arange(len(daily))

    for _ in range(args.bootstrap_samples):
        sampled_indices = rng.choice(
            daily_indices,
            size=len(daily_indices),
            replace=True,
        )

        sampled = daily.iloc[
            sampled_indices
        ]

        result = aggregate_sample(sampled)

        for key, value in result.items():
            bootstrap_results[key].append(value)

    bootstrap_summary = {
        "mae_delta": interval_summary(
            np.asarray(
                bootstrap_results["mae_delta"]
            ),
            better_direction="lower",
        ),
        "rmse_delta": interval_summary(
            np.asarray(
                bootstrap_results["rmse_delta"]
            ),
            better_direction="lower",
        ),
        "absolute_bias_delta": interval_summary(
            np.asarray(
                bootstrap_results["bias_abs_delta"]
            ),
            better_direction="lower",
        ),
        "ndcg5_delta": interval_summary(
            np.asarray(
                bootstrap_results["ndcg5_delta"]
            ),
            better_direction="higher",
        ),
        "ndcg10_delta": interval_summary(
            np.asarray(
                bootstrap_results["ndcg10_delta"]
            ),
            better_direction="higher",
        ),
        "top5_actual_k_delta": interval_summary(
            np.asarray(
                bootstrap_results[
                    "top5_actual_k_delta"
                ]
            ),
            better_direction="higher",
        ),
        "top5_regret_delta": interval_summary(
            np.asarray(
                bootstrap_results[
                    "top5_regret_delta"
                ]
            ),
            better_direction="lower",
        ),
    }

    overall_deltas = {
        "mae": (
            challenger_summary["mae"]
            - control_summary["mae"]
        ),
        "rmse": (
            challenger_summary["rmse"]
            - control_summary["rmse"]
        ),
        "bias": (
            challenger_summary["bias"]
            - control_summary["bias"]
        ),
    }

    date_win_rates = {
        "challenger_lower_mae_rate": float(
            (
                daily["challenger_mae"]
                < daily["control_mae"]
            ).mean()
        ),
        "challenger_lower_rmse_rate": float(
            (
                daily["challenger_rmse"]
                < daily["control_rmse"]
            ).mean()
        ),
        "challenger_higher_ndcg5_rate": float(
            (
                daily["challenger_ndcg5"]
                > daily["control_ndcg5"]
            ).mean()
        ),
        "challenger_lower_top5_regret_rate": float(
            (
                daily["challenger_top5_regret"]
                < daily["control_top5_regret"]
            ).mean()
        ),
    }

    monthly_win_counts = {
        "months": int(len(monthly)),
        "challenger_lower_mae_months": int(
            (
                monthly[
                    "mae_delta_challenger_minus_control"
                ] < 0
            ).sum()
        ),
        "challenger_lower_rmse_months": int(
            (
                monthly[
                    "rmse_delta_challenger_minus_control"
                ] < 0
            ).sum()
        ),
    }

    paired_path = out_dir / "paired_predictions.parquet"
    daily_path = out_dir / "daily_metrics.csv"
    monthly_path = out_dir / "monthly_metrics.csv"
    summary_path = out_dir / "summary.json"

    paired.to_parquet(
        paired_path,
        index=False,
    )

    daily.to_csv(
        daily_path,
        index=False,
    )

    monthly.to_csv(
        monthly_path,
        index=False,
    )

    summary = {
        "safety_version": SAFETY_VERSION,
        "predictions": str(predictions_path),
        "control": args.control,
        "challenger": args.challenger,
        "paired_keys": keys,
        "paired_rows": int(len(paired)),
        "paired_dates": int(
            paired["game_date"].nunique()
        ),
        "bootstrap_samples": int(
            args.bootstrap_samples
        ),
        "control_metrics": control_summary,
        "challenger_metrics": challenger_summary,
        "overall_deltas_challenger_minus_control": (
            overall_deltas
        ),
        "date_win_rates": date_win_rates,
        "monthly_win_counts": monthly_win_counts,
        "bootstrap": bootstrap_summary,
        "outputs": {
            "paired_predictions": str(paired_path),
            "daily_metrics": str(daily_path),
            "monthly_metrics": str(monthly_path),
            "summary": str(summary_path),
        },
        "interpretation": {
            "strong_evidence_mae": (
                bootstrap_summary[
                    "mae_delta"
                ]["ci_97_5"] < 0
            ),
            "strong_evidence_rmse": (
                bootstrap_summary[
                    "rmse_delta"
                ]["ci_97_5"] < 0
            ),
            "strong_evidence_ndcg5": (
                bootstrap_summary[
                    "ndcg5_delta"
                ]["ci_2_5"] > 0
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
