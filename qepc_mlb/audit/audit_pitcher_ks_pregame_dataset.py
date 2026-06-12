#!/usr/bin/env python3
"""
Audit the current QEPC-MLB pitcher strikeout dataset and legacy champion.

This script does not train a model.

It checks:
- Whether the old champion selected current-game outcome/workload columns
- Duplicate pitcher/game rows
- Multiple same-day appearances needing explicit ordering
- Suspicious feature names
- Timestamp and cutoff fields currently available
- Raw Savant fields requiring provenance review
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SAFETY_VERSION = "pitcher_ks_pregame_audit_v1"

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

SUSPICIOUS_NAME_PATTERN = re.compile(
    r"(same_game|target_game|current_game|postgame|future|"
    r"actual_|_actual|final_|_final|observed_|_observed|"
    r"next_game|next_start|closing_|result)",
    flags=re.IGNORECASE,
)

TIME_NAME_PATTERN = re.compile(
    r"(timestamp|time|scheduled|start_utc|start_time|"
    r"available_at|issued_at|published_at|cutoff|snapshot)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_enriched_opp_env_lineup.parquet"
        ),
    )
    parser.add_argument(
        "--feature_list",
        default=(
            "artifacts/mlb/models/"
            "pitcher_strikeouts_champion/feature_list.json"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/mlb/audits/pitcher_ks_pregame",
    )

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported data type: {path}")


def load_selected_features(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "numeric_features": [],
            "categorical_features": [],
        }

    obj = json.loads(path.read_text())

    return {
        "numeric_features": list(obj.get("numeric_features", [])),
        "categorical_features": list(
            obj.get("categorical_features", [])
        ),
    }


def stringify_examples(
    df: pd.DataFrame,
    columns: list[str],
    limit: int = 25,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    keep = [c for c in columns if c in df.columns]
    if not keep:
        return []

    return (
        df[keep]
        .head(limit)
        .astype("string")
        .fillna("")
        .to_dict(orient="records")
    )


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    feature_path = Path(args.feature_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {data_path}")
    df = read_table(data_path)

    selected_payload = load_selected_features(feature_path)

    selected_numeric = selected_payload["numeric_features"]
    selected_categorical = selected_payload[
        "categorical_features"
    ]
    selected_all = selected_numeric + selected_categorical

    selected_set = set(selected_all)
    column_set = set(df.columns)

    forbidden_present_in_data = sorted(
        column_set & FORBIDDEN_MODEL_FEATURES
    )
    forbidden_selected = sorted(
        selected_set & FORBIDDEN_MODEL_FEATURES
    )

    suspicious_columns = sorted(
        c for c in df.columns
        if SUSPICIOUS_NAME_PATTERN.search(c)
    )

    suspicious_selected = sorted(
        c for c in selected_all
        if SUSPICIOUS_NAME_PATTERN.search(c)
    )

    timestamp_columns = sorted(
        c for c in df.columns
        if TIME_NAME_PATTERN.search(c)
    )

    raw_savant_columns = sorted(
        c for c in df.columns
        if c.startswith("savant_")
    )

    selected_raw_savant_columns = sorted(
        c for c in selected_all
        if c.startswith("savant_")
    )

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(
            df["game_date"],
            errors="coerce",
        )

    # Prefer a true game ID key when available.
    if {"game_id", "pitcher_id"}.issubset(df.columns):
        primary_key = ["game_id", "pitcher_id"]
    elif {
        "game_date",
        "pitcher_id",
        "team",
        "opponent",
    }.issubset(df.columns):
        primary_key = [
            "game_date",
            "pitcher_id",
            "team",
            "opponent",
        ]
    else:
        primary_key = [
            c for c in ["game_date", "pitcher_id"]
            if c in df.columns
        ]

    if primary_key:
        duplicate_mask = df.duplicated(
            primary_key,
            keep=False,
        )
        duplicate_rows = df.loc[duplicate_mask].copy()
    else:
        duplicate_rows = pd.DataFrame()

    same_day_multiple = pd.DataFrame()

    if {"pitcher_id", "game_date"}.issubset(df.columns):
        same_day_counts = (
            df.groupby(
                ["pitcher_id", "game_date"],
                dropna=False,
            )
            .size()
            .rename("row_count")
            .reset_index()
        )

        same_day_multiple = same_day_counts[
            same_day_counts["row_count"] > 1
        ].copy()

    date_min = None
    date_max = None

    if "game_date" in df.columns:
        valid_dates = df["game_date"].dropna()
        if not valid_dates.empty:
            date_min = str(valid_dates.min())
            date_max = str(valid_dates.max())

    duplicate_csv = out_dir / "duplicate_key_rows.csv"
    same_day_csv = out_dir / "same_day_multiple_rows.csv"
    suspicious_csv = out_dir / "suspicious_columns.csv"
    report_path = out_dir / "audit_summary.json"

    if not duplicate_rows.empty:
        duplicate_rows.to_csv(
            duplicate_csv,
            index=False,
        )

    if not same_day_multiple.empty:
        same_day_multiple.to_csv(
            same_day_csv,
            index=False,
        )

    suspicious_frame = pd.DataFrame(
        {
            "column": sorted(
                set(suspicious_columns)
                | set(suspicious_selected)
            )
        }
    )

    if not suspicious_frame.empty:
        suspicious_frame["selected_by_old_champion"] = (
            suspicious_frame["column"].isin(selected_set)
        )
        suspicious_frame.to_csv(
            suspicious_csv,
            index=False,
        )

    status = "PASS_WITH_WARNINGS"

    if forbidden_selected:
        status = "FAIL_OLD_CHAMPION_LEAKAGE"

    report = {
        "safety_version": SAFETY_VERSION,
        "status": status,
        "data": str(data_path),
        "feature_list": str(feature_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": date_min,
        "date_max": date_max,
        "primary_key_used": primary_key,
        "duplicate_primary_key_rows": int(
            len(duplicate_rows)
        ),
        "same_day_pitcher_groups_with_multiple_rows": int(
            len(same_day_multiple)
        ),
        "selected_numeric_feature_count": int(
            len(selected_numeric)
        ),
        "selected_categorical_feature_count": int(
            len(selected_categorical)
        ),
        "forbidden_columns_present_in_data": (
            forbidden_present_in_data
        ),
        "forbidden_columns_selected_by_old_champion": (
            forbidden_selected
        ),
        "suspicious_columns_in_data": suspicious_columns,
        "suspicious_columns_selected_by_old_champion": (
            suspicious_selected
        ),
        "timestamp_or_cutoff_columns_found": timestamp_columns,
        "raw_savant_column_count": int(
            len(raw_savant_columns)
        ),
        "raw_savant_columns_selected_by_old_champion": (
            selected_raw_savant_columns
        ),
        "duplicate_examples": stringify_examples(
            duplicate_rows,
            [
                "game_id",
                "game_date",
                "pitcher_id",
                "pitcher_name",
                "team",
                "opponent",
                "is_starter",
            ],
        ),
        "same_day_multiple_examples": stringify_examples(
            same_day_multiple,
            [
                "pitcher_id",
                "game_date",
                "row_count",
            ],
        ),
        "notes": [
            (
                "Raw current-game outcome columns may remain in the "
                "training table because they are needed as targets or "
                "to build shifted historical features."
            ),
            (
                "They must never appear in the final pregame model "
                "feature list."
            ),
            (
                "Raw savant_* columns require provenance review before "
                "use. A season-level value may contain future games."
            ),
            (
                "Same-day multiple rows require scheduled timestamp "
                "and game-order auditing before rolling calculations."
            ),
        ],
        "outputs": {
            "summary": str(report_path),
            "duplicate_rows": (
                str(duplicate_csv)
                if not duplicate_rows.empty
                else None
            ),
            "same_day_multiple_rows": (
                str(same_day_csv)
                if not same_day_multiple.empty
                else None
            ),
            "suspicious_columns": (
                str(suspicious_csv)
                if not suspicious_frame.empty
                else None
            ),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, default=str)
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
