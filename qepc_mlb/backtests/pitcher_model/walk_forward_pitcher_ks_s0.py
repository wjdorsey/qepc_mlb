#!/usr/bin/env python3
"""
QEPC-MLB Pitcher Strikeouts S0 Walk-Forward Benchmark.

Uses only features explicitly listed in the S0 feature registry.

Models:
- Rolling prior-start strikeout baseline
- Prior BF × prior K/BF decomposed baseline
- HistGradientBoosting squared-error
- HistGradientBoosting Poisson
- Extra Trees

Important:
This evaluates actual starters. Operational announced-starter accuracy and
late scratches will be handled separately in the live scoring system.
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SAFETY_VERSION = "pitcher_ks_s0_walk_forward_v1"
TARGET = "strikeouts"

FORBIDDEN_FEATURES = {
    "strikeouts",
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "pitches",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
    "decision",
    "lineup_batting_ks",
    "lineup_batting_pa",
}


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
            "pitcher_strikeouts_pregame_s0"
        ),
    )

    parser.add_argument(
        "--test_start",
        default="2024-07-01",
        help="First date eligible for walk-forward scoring.",
    )
    parser.add_argument(
        "--test_end",
        default=None,
        help="Optional final scoring date.",
    )
    parser.add_argument(
        "--period",
        default="M",
        help="Pandas period code, normally M or W.",
    )
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
        "--models",
        nargs="*",
        default=[
            "hgb_squared",
            "hgb_poisson",
            "extra_trees",
        ],
        choices=[
            "hgb_squared",
            "hgb_poisson",
            "extra_trees",
        ],
    )

    parser.add_argument(
        "--hgb_max_iter",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--extra_trees_estimators",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
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


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input type: {path}")


def load_registry(
    path: Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text())

    numeric = list(payload.get("numeric_features", []))
    categorical = list(
        payload.get("categorical_features", [])
    )

    selected = set(numeric) | set(categorical)
    forbidden = sorted(selected & FORBIDDEN_FEATURES)

    if forbidden:
        raise RuntimeError(
            "Forbidden features found in registry: "
            f"{forbidden}"
        )

    raw_savant = sorted(
        c for c in selected
        if c.startswith("savant_")
    )

    if raw_savant:
        raise RuntimeError(
            "Raw Savant features found in S0 registry: "
            f"{raw_savant}"
        )

    return numeric, categorical, payload


def coerce_boolish(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()

    out = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False

    out.loc[
        out.isna()
        & text.isin(
            ["true", "t", "yes", "y", "starter", "sp"]
        )
    ] = True

    out.loc[
        out.isna()
        & text.isin(
            ["false", "f", "no", "n", "reliever", "rp"]
        )
    ] = False

    return out


def dense_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def prepare_x(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    columns = list(numeric) + list(categorical)
    x = frame[columns].copy()

    for column in numeric:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        ).astype(float)

    for column in categorical:
        x[column] = (
            x[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return x


def active_features(
    train: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    active_numeric: list[str] = []
    active_categorical: list[str] = []
    dropped: list[str] = []

    for column in numeric:
        values = pd.to_numeric(
            train[column],
            errors="coerce",
        )

        if values.notna().any():
            active_numeric.append(column)
        else:
            dropped.append(column)

    for column in categorical:
        values = train[column].astype("string")

        if values.notna().any():
            active_categorical.append(column)
        else:
            dropped.append(column)

    return (
        active_numeric,
        active_categorical,
        sorted(dropped),
    )


def build_pipeline(
    model_name: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    args: argparse.Namespace,
) -> Pipeline:
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
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

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if model_name == "hgb_squared":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=args.hgb_max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.50,
            random_state=args.random_state,
        )

    elif model_name == "hgb_poisson":
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.04,
            max_iter=args.hgb_max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.50,
            random_state=args.random_state,
        )

    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.extra_trees_estimators,
            max_depth=14,
            min_samples_leaf=8,
            max_features=1.0,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    residual = predicted - actual
    absolute_error = np.abs(residual)

    return {
        "rmse": float(
            math.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "median_absolute_error": float(
            np.median(absolute_error)
        ),
        "bias": float(np.mean(residual)),
        "median_bias": float(np.median(residual)),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
    }


def ranking_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float | int | None]:
    daily_spearman: list[float] = []
    daily_ndcg5: list[float] = []
    daily_ndcg10: list[float] = []
    daily_top5_actual_mean: list[float] = []
    daily_top10_actual_mean: list[float] = []
    daily_top5_regret: list[float] = []
    daily_top10_regret: list[float] = []

    eligible_dates = 0

    for _, day in predictions.groupby("game_date"):
        day = day.dropna(
            subset=[
                "actual_strikeouts",
                "predicted_strikeouts",
            ]
        ).copy()

        if len(day) < 2:
            continue

        eligible_dates += 1

        correlation = day[
            [
                "actual_strikeouts",
                "predicted_strikeouts",
            ]
        ].corr(method="spearman").iloc[0, 1]

        if pd.notna(correlation):
            daily_spearman.append(float(correlation))

        actual = day[
            "actual_strikeouts"
        ].to_numpy(dtype=float)

        predicted = day[
            "predicted_strikeouts"
        ].to_numpy(dtype=float)

        if actual.sum() > 0:
            k5 = min(5, len(day))
            k10 = min(10, len(day))

            daily_ndcg5.append(
                float(
                    ndcg_score(
                        actual.reshape(1, -1),
                        predicted.reshape(1, -1),
                        k=k5,
                    )
                )
            )

            daily_ndcg10.append(
                float(
                    ndcg_score(
                        actual.reshape(1, -1),
                        predicted.reshape(1, -1),
                        k=k10,
                    )
                )
            )

        predicted_order = day.sort_values(
            "predicted_strikeouts",
            ascending=False,
        )
        actual_order = day.sort_values(
            "actual_strikeouts",
            ascending=False,
        )

        for k, mean_store, regret_store in [
            (
                5,
                daily_top5_actual_mean,
                daily_top5_regret,
            ),
            (
                10,
                daily_top10_actual_mean,
                daily_top10_regret,
            ),
        ]:
            use_k = min(k, len(day))

            predicted_top = predicted_order.head(use_k)
            actual_top = actual_order.head(use_k)

            mean_store.append(
                float(
                    predicted_top[
                        "actual_strikeouts"
                    ].mean()
                )
            )

            regret_store.append(
                float(
                    actual_top[
                        "actual_strikeouts"
                    ].sum()
                    - predicted_top[
                        "actual_strikeouts"
                    ].sum()
                )
            )

    def safe_mean(values: list[float]) -> float | None:
        return (
            float(np.mean(values))
            if values
            else None
        )

    return {
        "ranking_dates": int(eligible_dates),
        "daily_spearman_mean": safe_mean(
            daily_spearman
        ),
        "daily_ndcg5_mean": safe_mean(
            daily_ndcg5
        ),
        "daily_ndcg10_mean": safe_mean(
            daily_ndcg10
        ),
        "daily_top5_actual_k_mean": safe_mean(
            daily_top5_actual_mean
        ),
        "daily_top10_actual_k_mean": safe_mean(
            daily_top10_actual_mean
        ),
        "daily_top5_regret_mean": safe_mean(
            daily_top5_regret
        ),
        "daily_top10_regret_mean": safe_mean(
            daily_top10_regret
        ),
    }


def prediction_frame(
    test: pd.DataFrame,
    predicted: np.ndarray,
    model_name: str,
    period: str,
) -> pd.DataFrame:
    metadata = [
        c for c in [
            "game_id",
            "game_date",
            "season",
            "pitcher_id",
            "pitcher_name",
            "team",
            "opponent",
        ]
        if c in test.columns
    ]

    out = test[metadata].copy()

    out["model"] = model_name
    out["period"] = period
    out["actual_strikeouts"] = pd.to_numeric(
        test[TARGET],
        errors="coerce",
    ).to_numpy()

    out["predicted_strikeouts"] = np.clip(
        np.asarray(predicted, dtype=float),
        0,
        None,
    )

    return out


def valid_test_periods(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[str]:
    test_start = pd.Timestamp(args.test_start)
    test_end = (
        pd.Timestamp(args.test_end)
        if args.test_end
        else None
    )

    periods = sorted(
        df["_period"].dropna().unique().tolist()
    )

    valid: list[str] = []

    for period in periods:
        test = df[df["_period"].eq(period)]

        if test.empty:
            continue

        period_min = test["game_date"].min()
        period_max = test["game_date"].max()

        if period_max < test_start:
            continue

        if test_end is not None and period_min > test_end:
            continue

        train = df[
            df["game_date"] < period_min
        ]

        if len(train) >= args.min_train_rows:
            valid.append(period)

    return valid


def baseline_predictions(
    df: pd.DataFrame,
    periods: Sequence[str],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    prediction_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []

    specifications = {
        "baseline_prior_start_k_roll5": (
            "prior_start_strikeouts_roll5_mean"
        ),
        "baseline_bf_x_k_rate_roll5": None,
    }

    for period in periods:
        test = df[df["_period"].eq(period)].copy()

        for model_name, source_column in (
            specifications.items()
        ):
            if source_column is not None:
                if source_column not in test.columns:
                    continue

                predicted = pd.to_numeric(
                    test[source_column],
                    errors="coerce",
                )

            else:
                needed = [
                    "prior_start_batters_faced_roll5_mean",
                    "prior_start_k_per_bf_roll5",
                ]

                if not all(
                    c in test.columns
                    for c in needed
                ):
                    continue

                predicted = (
                    pd.to_numeric(
                        test[
                            "prior_start_batters_faced_roll5_mean"
                        ],
                        errors="coerce",
                    )
                    * pd.to_numeric(
                        test[
                            "prior_start_k_per_bf_roll5"
                        ],
                        errors="coerce",
                    )
                )

            valid_mask = predicted.notna()

            if not valid_mask.any():
                continue

            scored_test = test.loc[
                valid_mask
            ].copy()

            scored_predictions = predicted.loc[
                valid_mask
            ].to_numpy(dtype=float)

            frame = prediction_frame(
                scored_test,
                scored_predictions,
                model_name,
                period,
            )

            prediction_frames.append(frame)

            metrics = point_metrics(
                frame["actual_strikeouts"],
                frame["predicted_strikeouts"],
            )

            window_rows.append(
                {
                    "model": model_name,
                    "period": period,
                    "train_rows": None,
                    "test_rows": int(len(frame)),
                    "feature_count": (
                        1
                        if source_column is not None
                        else 2
                    ),
                    **metrics,
                }
            )

    return prediction_frames, window_rows


def model_predictions(
    df: pd.DataFrame,
    periods: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_name: str,
    args: argparse.Namespace,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    prediction_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []

    iterator = periods

    if tqdm is not None and not args.no_progress:
        iterator = tqdm(
            periods,
            desc=f"Walk-forward {model_name}",
            unit="period",
        )

    for period in iterator:
        test = df[df["_period"].eq(period)].copy()

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

        pipeline = build_pipeline(
            model_name,
            use_numeric,
            use_categorical,
            args,
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

        y_train = pd.to_numeric(
            train[TARGET],
            errors="coerce",
        ).astype(float)

        pipeline.fit(x_train, y_train)

        predicted = pipeline.predict(x_test)

        frame = prediction_frame(
            test,
            predicted,
            model_name,
            period,
        )

        prediction_frames.append(frame)

        metrics = point_metrics(
            frame["actual_strikeouts"],
            frame["predicted_strikeouts"],
        )

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
                "numeric_feature_count": int(
                    len(use_numeric)
                ),
                "categorical_feature_count": int(
                    len(use_categorical)
                ),
                "dropped_empty_feature_count": int(
                    len(dropped)
                ),
                **metrics,
            }
        )

    return prediction_frames, window_rows


def summarize_model(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    point = point_metrics(
        predictions["actual_strikeouts"],
        predictions["predicted_strikeouts"],
    )

    ranking = ranking_metrics(predictions)

    return {
        **point,
        **ranking,
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

    missing_features = sorted(
        (
            set(numeric_features)
            | set(categorical_features)
        )
        - set(df.columns)
    )

    if missing_features:
        raise RuntimeError(
            "Registry features missing from dataset: "
            f"{missing_features}"
        )

    required = [
        "game_date",
        "pitcher_id",
        TARGET,
        "is_starter",
    ]

    missing_required = [
        c for c in required
        if c not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df["is_starter"] = coerce_boolish(
        df["is_starter"]
    )

    df = df[
        df["game_date"].notna()
        & df[TARGET].notna()
        & df["is_starter"].fillna(False).astype(bool)
    ].copy()

    prior_start_count_column = (
        "prior_start_dates_prior"
    )

    if prior_start_count_column in df.columns:
        prior_starts = pd.to_numeric(
            df[prior_start_count_column],
            errors="coerce",
        ).fillna(0)

        df = df[
            prior_starts >= args.min_prior_starts
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
            "No valid walk-forward periods. "
            "Lower --min_train_rows or move --test_start."
        )

    print(
        f"Eligible starter rows: {len(df):,}"
    )
    print(
        f"Walk-forward periods: {len(periods)}"
    )
    print(
        f"First period: {periods[0]}"
    )
    print(
        f"Last period: {periods[-1]}"
    )
    print(
        "Registry features: "
        f"{len(numeric_features)} numeric + "
        f"{len(categorical_features)} categorical"
    )

    all_prediction_frames: list[pd.DataFrame] = []
    all_window_rows: list[dict[str, Any]] = []

    baseline_frames, baseline_windows = (
        baseline_predictions(
            df,
            periods,
        )
    )

    all_prediction_frames.extend(
        baseline_frames
    )
    all_window_rows.extend(
        baseline_windows
    )

    for model_name in args.models:
        frames, windows = model_predictions(
            df,
            periods,
            numeric_features,
            categorical_features,
            model_name,
            args,
        )

        all_prediction_frames.extend(frames)
        all_window_rows.extend(windows)

    if not all_prediction_frames:
        raise RuntimeError(
            "No predictions were generated."
        )

    predictions = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    window_metrics = pd.DataFrame(
        all_window_rows
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

    best_top10 = (
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

    best_top10.to_csv(
        top10_path,
        index=False,
    )

    if args.also_csv:
        predictions.to_csv(
            out_dir / "predictions.csv",
            index=False,
        )

    final_summary = {
        "safety_version": SAFETY_VERSION,
        "input": str(input_path),
        "registry": str(registry_path),
        "registry_safety_version": (
            registry_payload.get("safety_version")
        ),
        "evaluation_population": (
            "Actual starting pitchers with the required "
            "number of prior starter dates."
        ),
        "test_start": args.test_start,
        "test_end": args.test_end,
        "period": args.period,
        "min_prior_starts": args.min_prior_starts,
        "min_train_rows": args.min_train_rows,
        "eligible_rows": int(len(df)),
        "walk_forward_periods": periods,
        "numeric_feature_count": int(
            len(numeric_features)
        ),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "forbidden_features_selected": [],
        "models_tested": [
            "baseline_prior_start_k_roll5",
            "baseline_bf_x_k_rate_roll5",
            *args.models,
        ],
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
            "S0 is an intentionally conservative pregame "
            "baseline. It excludes opponent, confirmed-lineup, "
            "raw Savant, and unverified weather features."
        ),
    }

    summary_path.write_text(
        json.dumps(
            final_summary,
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
                "outputs": final_summary["outputs"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
