#!/usr/bin/env python3
"""
Build the QEPC-MLB pitcher strikeout Pregame-S0 feature dataset.

S0 is the honest audit baseline.

Safety rules:
- Target-game outcomes are retained only as labels/metadata.
- Every rolling feature uses dates strictly before the target date.
- Multiple appearances on the same date are aggregated for future history.
- Same-day appearances never feed one another.
- Raw Savant, lineup, opponent-roll, and observed-weather fields are excluded.
- Only an explicit feature registry is written for model use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SAFETY_VERSION = "pitcher_ks_pregame_s0_feature_build_v1"

FORBIDDEN_MODEL_FEATURES = {
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

HISTORY_COLUMNS = [
    "strikeouts",
    "batters_faced",
    "outs_recorded",
    "pitches",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
]

STATIC_CATEGORICAL_CANDIDATES = [
    "pitcher_hand",
    "team",
    "opponent",
    "home_away",
    "env_park_id",
    "env_daynight",
    "env_usedh",
]

METADATA_CANDIDATES = [
    "game_id",
    "game_date",
    "season",
    "pitcher_id",
    "pitcher_name",
    "pitcher_key",
    "team",
    "opponent",
    "is_starter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_enriched_opp_env_lineup.parquet"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s0.parquet"
        ),
    )
    parser.add_argument("--also_csv", action="store_true")

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input type: {path}")


def coerce_boolish(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()

    out = pd.Series(pd.NA, index=series.index, dtype="boolean")

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


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def rolling_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    window: int,
) -> pd.Series:
    numerator_prior = numerator.shift(1)
    denominator_prior = denominator.shift(1)

    numerator_sum = numerator_prior.rolling(
        window,
        min_periods=1,
    ).sum()

    denominator_sum = denominator_prior.rolling(
        window,
        min_periods=1,
    ).sum()

    return numerator_sum / denominator_sum.replace(0, np.nan)


def build_daily_history(
    source: pd.DataFrame,
    prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate one pitcher/date row and create shifted history.

    Every row for a pitcher on a target date receives features built
    strictly from earlier dates.
    """
    available_history = [
        c for c in HISTORY_COLUMNS
        if c in source.columns
    ]

    aggregations = {
        c: "sum"
        for c in available_history
    }

    daily = (
        source.groupby(
            ["pitcher_id", "game_date"],
            as_index=False,
            dropna=False,
        )
        .agg(aggregations)
        .sort_values(
            ["pitcher_id", "game_date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    feature_names: list[str] = []
    output_parts: list[pd.DataFrame] = []

    for _, group in daily.groupby(
        "pitcher_id",
        sort=False,
    ):
        group = group.copy()
        group = group.sort_values(
            "game_date",
            kind="mergesort",
        )

        features = group[
            ["pitcher_id", "game_date"]
        ].copy()

        features[f"{prefix}_dates_prior"] = np.arange(
            len(group)
        )

        features[f"{prefix}_days_since_prev"] = (
            group["game_date"]
            - group["game_date"].shift(1)
        ).dt.days

        feature_names.extend(
            [
                f"{prefix}_dates_prior",
                f"{prefix}_days_since_prev",
            ]
        )

        for column in available_history:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            lag_name = f"{prefix}_{column}_lag1"
            features[lag_name] = values.shift(1)
            feature_names.append(lag_name)

            for window in [3, 5, 10]:
                mean_name = (
                    f"{prefix}_{column}_roll"
                    f"{window}_mean"
                )
                features[mean_name] = (
                    values.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
                feature_names.append(mean_name)

        if {
            "strikeouts",
            "batters_faced",
        }.issubset(available_history):
            strikeouts = pd.to_numeric(
                group["strikeouts"],
                errors="coerce",
            )
            batters_faced = pd.to_numeric(
                group["batters_faced"],
                errors="coerce",
            )

            for window in [3, 5, 10]:
                name = (
                    f"{prefix}_k_per_bf_roll"
                    f"{window}"
                )
                features[name] = rolling_ratio(
                    strikeouts,
                    batters_faced,
                    window,
                )
                feature_names.append(name)

        if {
            "walks",
            "batters_faced",
        }.issubset(available_history):
            walks = pd.to_numeric(
                group["walks"],
                errors="coerce",
            )
            batters_faced = pd.to_numeric(
                group["batters_faced"],
                errors="coerce",
            )

            for window in [5, 10]:
                name = (
                    f"{prefix}_bb_per_bf_roll"
                    f"{window}"
                )
                features[name] = rolling_ratio(
                    walks,
                    batters_faced,
                    window,
                )
                feature_names.append(name)

        if {
            "outs_recorded",
            "batters_faced",
        }.issubset(available_history):
            outs = pd.to_numeric(
                group["outs_recorded"],
                errors="coerce",
            )
            batters_faced = pd.to_numeric(
                group["batters_faced"],
                errors="coerce",
            )

            for window in [5, 10]:
                name = (
                    f"{prefix}_outs_per_bf_roll"
                    f"{window}"
                )
                features[name] = rolling_ratio(
                    outs,
                    batters_faced,
                    window,
                )
                feature_names.append(name)

        output_parts.append(features)

    output = pd.concat(
        output_parts,
        ignore_index=True,
    )

    feature_names = dedupe(feature_names)

    return output, feature_names


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    raw = read_table(input_path)
    raw_rows = int(len(raw))

    required = [
        "game_date",
        "pitcher_id",
        "strikeouts",
    ]
    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = raw.copy()

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df = df[
        df["game_date"].notna()
        & df["pitcher_id"].notna()
    ].copy()

    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    if "is_starter" in df.columns:
        df["is_starter"] = coerce_boolish(
            df["is_starter"]
        )
    else:
        df["is_starter"] = pd.Series(
            False,
            index=df.index,
            dtype="boolean",
        )

    for column in HISTORY_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if {"game_id", "pitcher_id"}.issubset(df.columns):
        duplicate_count = int(
            df.duplicated(
                ["game_id", "pitcher_id"],
            ).sum()
        )

        if duplicate_count:
            raise RuntimeError(
                "Duplicate game_id/pitcher_id rows found "
                f"before feature building: {duplicate_count}"
            )
    else:
        duplicate_count = 0

    # All-appearance history.
    all_history, all_features = build_daily_history(
        df,
        prefix="all_app",
    )

    # Starter-only history.
    starters = df[
        df["is_starter"].fillna(False).astype(bool)
    ].copy()

    starter_history, starter_features = (
        build_daily_history(
            starters,
            prefix="prior_start",
        )
        if not starters.empty
        else (
            pd.DataFrame(
                columns=["pitcher_id", "game_date"]
            ),
            [],
        )
    )

    metadata_columns = [
        c for c in METADATA_CANDIDATES
        if c in df.columns
    ]

    target_columns = [
        c for c in [
            "strikeouts",
            "batters_faced",
            "outs_recorded",
            "pitches",
        ]
        if c in df.columns
    ]

    base_columns = dedupe(
        metadata_columns + target_columns
    )

    output = df[base_columns].copy()

    output = output.merge(
        all_history,
        how="left",
        on=["pitcher_id", "game_date"],
        validate="many_to_one",
    )

    if not starter_history.empty:
        output = output.merge(
            starter_history,
            how="left",
            on=["pitcher_id", "game_date"],
            validate="many_to_one",
        )

    categorical_features = [
        c for c in STATIC_CATEGORICAL_CANDIDATES
        if c in df.columns
    ]

    # Add only explicitly approved static categorical fields.
    for column in categorical_features:
        if column not in output.columns:
            static_map = (
                df[
                    [
                        "pitcher_id",
                        "game_date",
                        column,
                    ]
                ]
                .drop_duplicates(
                    ["pitcher_id", "game_date"],
                    keep="first",
                )
            )

            output = output.merge(
                static_map,
                how="left",
                on=["pitcher_id", "game_date"],
                validate="many_to_one",
            )

    numeric_features = dedupe(
        all_features + starter_features
    )

    # Never permit target-game raw outcomes into the model registry.
    bad_features = sorted(
        (
            set(numeric_features)
            | set(categorical_features)
        )
        & FORBIDDEN_MODEL_FEATURES
    )

    if bad_features:
        raise RuntimeError(
            "Forbidden model features entered S0 registry: "
            f"{bad_features}"
        )

    # Confirm target-game mutation cannot affect same-row features:
    # all generated features are date-grouped and shifted before merging.
    same_day_groups = (
        df.groupby(
            ["pitcher_id", "game_date"],
            dropna=False,
        )
        .size()
        .gt(1)
        .sum()
    )

    registry = {
        "safety_version": SAFETY_VERSION,
        "prediction_contract": (
            "All rolling features use pitcher dates strictly "
            "before the target date."
        ),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "forbidden_features": sorted(
            FORBIDDEN_MODEL_FEATURES
        ),
        "excluded_feature_families": [
            "raw target-game outcomes",
            "raw savant season aggregates",
            "precomputed opponent rolling features",
            "reconstructed current-lineup features",
            "observed or unverified weather fields",
        ],
    }

    registry_path = output_path.with_suffix(
        ".feature_registry.json"
    )
    summary_path = output_path.with_suffix(
        ".summary.json"
    )

    output.to_parquet(
        output_path,
        index=False,
    )

    csv_path = None
    if args.also_csv:
        csv_path = output_path.with_suffix(".csv")
        output.to_csv(
            csv_path,
            index=False,
        )

    registry_path.write_text(
        json.dumps(
            registry,
            indent=2,
            default=str,
        )
    )

    summary = {
        "safety_version": SAFETY_VERSION,
        "input": str(input_path),
        "out": str(output_path),
        "csv": (
            str(csv_path)
            if csv_path is not None
            else None
        ),
        "feature_registry": str(registry_path),
        "raw_rows": raw_rows,
        "rows_written": int(len(output)),
        "columns_written": int(len(output.columns)),
        "duplicate_game_pitcher_rows": duplicate_count,
        "same_day_pitcher_groups_handled_as_date_batches": int(
            same_day_groups
        ),
        "starter_rows": int(
            output["is_starter"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "is_starter" in output.columns
        else None,
        "numeric_feature_count": int(
            len(numeric_features)
        ),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "forbidden_features_selected": bad_features,
        "raw_savant_features_selected": [],
        "lineup_features_selected": [],
        "unverified_weather_features_selected": [],
        "note": (
            "S0 is intentionally conservative. It establishes "
            "an uncontaminated pitcher-history baseline before "
            "opponent, lineup, Statcast, and forecast features "
            "are reintroduced through audited pipelines."
        ),
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
