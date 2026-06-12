#!/usr/bin/env python3
"""
QEPC-MLB Pitcher Strikeouts S1 Decomposed Walk-Forward Test.

Tests:
    Total strikeouts = predicted batters faced × predicted K/BF

All models use only the explicit S0 pregame-safe feature registry.

Outputs:
- Direct HGB strikeout control
- BF squared × raw-rate model
- BF Poisson × raw-rate model
- BF squared × logit-rate model
- BF Poisson × logit-rate model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# Reuse the already validated utility functions from S0.
from walk_forward_pitcher_ks_s0 import (
    active_features,
    coerce_boolish,
    dense_one_hot,
    load_registry,
    point_metrics,
    prediction_frame,
    prepare_x,
    ranking_metrics,
    read_table,
    valid_test_periods,
)


SAFETY_VERSION = "pitcher_ks_s1_decomposed_walk_forward_v1"

TARGET_K = "strikeouts"
TARGET_BF = "batters_faced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s0.parquet"
        ),
    )
    parser.add_argument(
        "--registry",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s0.feature_registry.json"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "artifacts/mlb/backtests/"
            "pitcher_strikeouts_pregame_s1_decomposed"
        ),
    )

    parser.add_argument("--test_start", default="2024-07-01")
    parser.add_argument("--test_end", default=None)
    parser.add_argument("--period", default="M")

    parser.add_argument(
        "--min_prior_starts",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min_train_rows",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--max_leaf_nodes",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--also_csv",
        action="store_true",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
    )

    return parser.parse_args()


def make_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> ColumnTransformer:
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        )
                    ]
                ),
                list(numeric),
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            dense_one_hot(),
                        ),
                    ]
                ),
                list(categorical),
            )
        )

    if not transformers:
        raise ValueError("No active features in fold.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_hgb(
    loss: str,
    args: argparse.Namespace,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )


def expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -20.0, 20.0)

    return 1.0 / (1.0 + np.exp(-values))


def empirical_logit_rate(
    strikeouts: np.ndarray,
    batters_faced: np.ndarray,
) -> np.ndarray:
    """
    Empirical-logit transform with 0.5 pseudo-counts.

    This keeps zero-K and all-K observations finite.
    """
    strikeouts = np.asarray(strikeouts, dtype=float)
    batters_faced = np.asarray(batters_faced, dtype=float)

    probability = (
        strikeouts + 0.5
    ) / (
        batters_faced + 1.0
    )

    probability = np.clip(
        probability,
        0.001,
        0.999,
    )

    return np.log(
        probability / (1.0 - probability)
    )


def component_metrics(
    frame: pd.DataFrame,
) -> dict[str, float | None]:
    if (
        "predicted_batters_faced" not in frame.columns
        or frame["predicted_batters_faced"].isna().all()
    ):
        return {
            "bf_rmse": None,
            "bf_mae": None,
            "bf_bias": None,
            "rate_mae": None,
            "rate_bf_weighted_mae": None,
        }

    actual_bf = frame[
        "actual_batters_faced"
    ].to_numpy(dtype=float)

    predicted_bf = frame[
        "predicted_batters_faced"
    ].to_numpy(dtype=float)

    actual_rate = frame[
        "actual_k_per_bf"
    ].to_numpy(dtype=float)

    predicted_rate = frame[
        "predicted_k_per_bf"
    ].to_numpy(dtype=float)

    rate_error = np.abs(
        predicted_rate - actual_rate
    )

    return {
        "bf_rmse": float(
            math.sqrt(
                mean_squared_error(
                    actual_bf,
                    predicted_bf,
                )
            )
        ),
        "bf_mae": float(
            mean_absolute_error(
                actual_bf,
                predicted_bf,
            )
        ),
        "bf_bias": float(
            np.mean(predicted_bf - actual_bf)
        ),
        "rate_mae": float(
            np.mean(rate_error)
        ),
        "rate_bf_weighted_mae": float(
            np.average(
                rate_error,
                weights=actual_bf,
            )
        ),
    }


def build_prediction_output(
    test: pd.DataFrame,
    predicted_k: np.ndarray,
    model_name: str,
    period: str,
    predicted_bf: np.ndarray | None = None,
    predicted_rate: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = prediction_frame(
        test=test,
        predicted=predicted_k,
        model_name=model_name,
        period=period,
    )

    actual_bf = pd.to_numeric(
        test[TARGET_BF],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual_k = pd.to_numeric(
        test[TARGET_K],
        errors="coerce",
    ).to_numpy(dtype=float)

    frame["actual_batters_faced"] = actual_bf
    frame["actual_k_per_bf"] = (
        actual_k / actual_bf
    )

    if predicted_bf is None:
        frame["predicted_batters_faced"] = np.nan
    else:
        frame["predicted_batters_faced"] = (
            np.asarray(predicted_bf, dtype=float)
        )

    if predicted_rate is None:
        frame["predicted_k_per_bf"] = np.nan
    else:
        frame["predicted_k_per_bf"] = (
            np.asarray(predicted_rate, dtype=float)
        )

    return frame


def summarize_model(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        **point_metrics(
            predictions["actual_strikeouts"],
            predictions["predicted_strikeouts"],
        ),
        **ranking_metrics(predictions),
        **component_metrics(predictions),
        "rows_scored": int(len(predictions)),
        "windows": int(
            predictions["period"].nunique()
        ),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading S0 data: {input_path}")

    df = read_table(input_path)

    (
        numeric_features,
        categorical_features,
        registry_payload,
    ) = load_registry(registry_path)

    required = [
        "game_date",
        "pitcher_id",
        "is_starter",
        TARGET_K,
        TARGET_BF,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df[TARGET_K] = pd.to_numeric(
        df[TARGET_K],
        errors="coerce",
    )

    df[TARGET_BF] = pd.to_numeric(
        df[TARGET_BF],
        errors="coerce",
    )

    df["is_starter"] = coerce_boolish(
        df["is_starter"]
    )

    df = df[
        df["game_date"].notna()
        & df[TARGET_K].notna()
        & df[TARGET_BF].notna()
        & (df[TARGET_BF] > 0)
        & df["is_starter"].fillna(False).astype(bool)
    ].copy()

    prior_count_column = "prior_start_dates_prior"

    if prior_count_column in df.columns:
        prior_count = pd.to_numeric(
            df[prior_count_column],
            errors="coerce",
        ).fillna(0)

        df = df[
            prior_count >= args.min_prior_starts
        ].copy()

    df = df.sort_values(
        ["game_date", "pitcher_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df["_period"] = (
        df["game_date"]
        .dt.to_period(args.period)
        .astype(str)
    )

    periods = valid_test_periods(df, args)

    if not periods:
        raise RuntimeError(
            "No valid walk-forward periods."
        )

    print(f"Eligible starter rows: {len(df):,}")
    print(f"Walk-forward periods: {len(periods)}")
    print(f"First period: {periods[0]}")
    print(f"Last period: {periods[-1]}")
    print(
        "Registry features: "
        f"{len(numeric_features)} numeric + "
        f"{len(categorical_features)} categorical"
    )

    prediction_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []

    iterator = periods

    if not args.no_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                periods,
                desc="Walk-forward S1 decomposition",
                unit="period",
            )
        except Exception:
            pass

    for period in iterator:
        test = df[
            df["_period"].eq(period)
        ].copy()

        if test.empty:
            continue

        period_start = test["game_date"].min()

        train = df[
            df["game_date"] < period_start
        ].copy()

        if len(train) < args.min_train_rows:
            continue

        (
            use_numeric,
            use_categorical,
            dropped,
        ) = active_features(
            train,
            numeric_features,
            categorical_features,
        )

        x_train = prepare_x(
            train,
            use_numeric,
            use_categorical,
        )

        x_test = prepare_x(
            test,
            use_numeric,
            use_categorical,
        )

        preprocessor = make_preprocessor(
            use_numeric,
            use_categorical,
        )

        train_matrix = preprocessor.fit_transform(
            x_train
        )

        test_matrix = preprocessor.transform(
            x_test
        )

        train_k = pd.to_numeric(
            train[TARGET_K],
            errors="coerce",
        ).to_numpy(dtype=float)

        test_k = pd.to_numeric(
            test[TARGET_K],
            errors="coerce",
        ).to_numpy(dtype=float)

        train_bf = pd.to_numeric(
            train[TARGET_BF],
            errors="coerce",
        ).to_numpy(dtype=float)

        test_bf = pd.to_numeric(
            test[TARGET_BF],
            errors="coerce",
        ).to_numpy(dtype=float)

        train_rate = np.clip(
            train_k / train_bf,
            0.0,
            0.70,
        )

        train_rate_logit = empirical_logit_rate(
            train_k,
            train_bf,
        )

        # ---------------------------------------------------------
        # Direct control
        # ---------------------------------------------------------
        direct_model = make_hgb(
            "squared_error",
            args,
        )

        direct_model.fit(
            train_matrix,
            train_k,
        )

        direct_prediction = np.clip(
            direct_model.predict(test_matrix),
            0.0,
            None,
        )

        direct_frame = build_prediction_output(
            test=test,
            predicted_k=direct_prediction,
            model_name="direct_hgb_squared",
            period=period,
        )

        prediction_frames.append(direct_frame)

        direct_metrics = point_metrics(
            direct_frame["actual_strikeouts"],
            direct_frame["predicted_strikeouts"],
        )

        window_rows.append(
            {
                "model": "direct_hgb_squared",
                "period": period,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "feature_count": int(
                    len(use_numeric)
                    + len(use_categorical)
                ),
                "dropped_empty_feature_count": int(
                    len(dropped)
                ),
                **direct_metrics,
            }
        )

        # ---------------------------------------------------------
        # Workload models
        # ---------------------------------------------------------
        bf_squared_model = make_hgb(
            "squared_error",
            args,
        )

        bf_squared_model.fit(
            train_matrix,
            train_bf,
        )

        predicted_bf_squared = np.clip(
            bf_squared_model.predict(test_matrix),
            1.0,
            40.0,
        )

        bf_poisson_model = make_hgb(
            "poisson",
            args,
        )

        bf_poisson_model.fit(
            train_matrix,
            train_bf,
        )

        predicted_bf_poisson = np.clip(
            bf_poisson_model.predict(test_matrix),
            1.0,
            40.0,
        )

        # ---------------------------------------------------------
        # Strikeout-rate models
        # ---------------------------------------------------------
        rate_raw_model = make_hgb(
            "squared_error",
            args,
        )

        rate_raw_model.fit(
            train_matrix,
            train_rate,
            sample_weight=train_bf,
        )

        predicted_rate_raw = np.clip(
            rate_raw_model.predict(test_matrix),
            0.01,
            0.60,
        )

        rate_logit_model = make_hgb(
            "squared_error",
            args,
        )

        rate_logit_model.fit(
            train_matrix,
            train_rate_logit,
            sample_weight=train_bf,
        )

        predicted_rate_logit = np.clip(
            expit(
                rate_logit_model.predict(
                    test_matrix
                )
            ),
            0.01,
            0.60,
        )

        combinations = [
            (
                "decomp_bf_squared_rate_raw",
                predicted_bf_squared,
                predicted_rate_raw,
            ),
            (
                "decomp_bf_poisson_rate_raw",
                predicted_bf_poisson,
                predicted_rate_raw,
            ),
            (
                "decomp_bf_squared_rate_logit",
                predicted_bf_squared,
                predicted_rate_logit,
            ),
            (
                "decomp_bf_poisson_rate_logit",
                predicted_bf_poisson,
                predicted_rate_logit,
            ),
        ]

        for (
            model_name,
            predicted_bf,
            predicted_rate,
        ) in combinations:
            predicted_k = (
                predicted_bf * predicted_rate
            )

            frame = build_prediction_output(
                test=test,
                predicted_k=predicted_k,
                model_name=model_name,
                period=period,
                predicted_bf=predicted_bf,
                predicted_rate=predicted_rate,
            )

            prediction_frames.append(frame)

            metrics = {
                **point_metrics(
                    frame["actual_strikeouts"],
                    frame["predicted_strikeouts"],
                ),
                **component_metrics(frame),
            }

            window_rows.append(
                {
                    "model": model_name,
                    "period": period,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "feature_count": int(
                        len(use_numeric)
                        + len(use_categorical)
                    ),
                    "dropped_empty_feature_count": int(
                        len(dropped)
                    ),
                    **metrics,
                }
            )

    if not prediction_frames:
        raise RuntimeError(
            "No walk-forward predictions generated."
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    window_metrics = pd.DataFrame(
        window_rows
    )

    summaries: list[dict[str, Any]] = []

    for model_name, group in predictions.groupby(
        "model",
        sort=False,
    ):
        summary = summarize_model(group)
        summary["model"] = model_name
        summaries.append(summary)

    ranking_by_mae = sorted(
        summaries,
        key=lambda row: (
            row["mae"],
            row["rmse"],
        ),
    )

    best_model = ranking_by_mae[0]["model"]

    best_predictions = predictions[
        predictions["model"].eq(best_model)
    ].copy()

    top10 = (
        best_predictions.sort_values(
            [
                "game_date",
                "predicted_strikeouts",
            ],
            ascending=[True, False],
        )
        .groupby(
            "game_date",
            group_keys=False,
        )
        .head(10)
        .copy()
    )

    predictions_path = (
        out_dir / "predictions.parquet"
    )
    windows_path = (
        out_dir / "window_metrics.csv"
    )
    top10_path = (
        out_dir / "best_model_top10_daily_board.csv"
    )
    summary_path = out_dir / "summary.json"

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    window_metrics.to_csv(
        windows_path,
        index=False,
    )

    top10.to_csv(
        top10_path,
        index=False,
    )

    if args.also_csv:
        predictions.to_csv(
            out_dir / "predictions.csv",
            index=False,
        )

    final = {
        "safety_version": SAFETY_VERSION,
        "input": str(input_path),
        "registry": str(registry_path),
        "registry_safety_version": (
            registry_payload.get("safety_version")
        ),
        "eligible_rows": int(len(df)),
        "walk_forward_periods": periods,
        "models_tested": sorted(
            predictions["model"].unique().tolist()
        ),
        "ranking_by_mae": ranking_by_mae,
        "best_model": ranking_by_mae[0],
        "outputs": {
            "predictions": str(predictions_path),
            "window_metrics": str(windows_path),
            "best_model_top10_daily_board": str(
                top10_path
            ),
            "summary": str(summary_path),
        },
        "note": (
            "S1 tests genuinely independent workload and "
            "strikeout-rate models using only S0 pregame-safe "
            "features."
        ),
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
            {
                "safety_version": SAFETY_VERSION,
                "eligible_rows": int(len(df)),
                "walk_forward_period_count": int(
                    len(periods)
                ),
                "ranking_by_mae": ranking_by_mae,
                "best_model": ranking_by_mae[0],
                "outputs": final["outputs"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
