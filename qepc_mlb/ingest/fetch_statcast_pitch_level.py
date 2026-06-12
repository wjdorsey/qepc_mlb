#!/usr/bin/env python3
"""
Download pitch-level Statcast data in resumable monthly chunks.

Features:
- Enables pybaseball's local cache
- Saves one atomic Parquet file per date chunk
- Skips already completed chunks
- Retries failed requests
- Validates important pitch-level columns
- Writes a cumulative manifest and JSON summary

This script only downloads raw data. It does not create model features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from pybaseball import cache, statcast


SAFETY_VERSION = "fetch_statcast_pitch_level_v1_resumable"

EXPECTED_COLUMNS = {
    "game_date",
    "game_pk",
    "pitcher",
    "batter",
    "pitch_type",
    "release_speed",
    "description",
    "events",
    "zone",
    "type",
    "balls",
    "strikes",
    "at_bat_number",
    "pitch_number",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True,
        help="First date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Last date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--out_dir",
        default="cache/mlb/raw/statcast_pitch_level",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow pybaseball to parallelize its requests. "
            "The default is off for maximum stability."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def monthly_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    cursor = start.replace(day=1)

    while cursor <= end:
        next_month = cursor + pd.offsets.MonthBegin(1)

        chunk_start = max(start, cursor)
        chunk_end = min(
            end,
            next_month - pd.Timedelta(days=1),
        )

        chunks.append((chunk_start, chunk_end))
        cursor = next_month

    return chunks


def chunk_filename(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    return (
        f"statcast_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    )


def parquet_is_readable(path: Path) -> tuple[bool, int | None]:
    if not path.exists():
        return False, None

    try:
        parquet = pq.ParquetFile(path)
        return True, int(parquet.metadata.num_rows)
    except Exception:
        return False, None


def make_parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type object columns to pandas strings when needed.

    Baseball Savant occasionally returns mixed Python types in one
    column, which can upset Parquet serialization.
    """
    frame = frame.copy()

    for column in frame.columns:
        if frame[column].dtype != "object":
            continue

        nonmissing = frame[column].dropna()

        if nonmissing.empty:
            continue

        type_count = nonmissing.map(type).nunique()

        if type_count > 1:
            frame[column] = frame[column].astype("string")

    return frame


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(
            frame["game_date"],
            errors="coerce",
        )

    sort_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
        if column in frame.columns
    ]

    if sort_columns:
        frame = frame.sort_values(
            sort_columns,
            kind="mergesort",
        )

    pitch_key = [
        column
        for column in [
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "pitcher",
            "batter",
        ]
        if column in frame.columns
    ]

    if len(pitch_key) >= 3:
        frame = frame.drop_duplicates(
            pitch_key,
            keep="last",
        )

    return make_parquet_safe(frame)


def fetch_chunk(
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int,
    retry_sleep: float,
    parallel: bool,
) -> pd.DataFrame:
    start_text = start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"Fetching {start_text} through {end_text} "
                f"(attempt {attempt}/{retries})..."
            )

            frame = statcast(
                start_dt=start_text,
                end_dt=end_text,
                verbose=True,
                parallel=parallel,
            )

            if frame is None:
                frame = pd.DataFrame()

            return normalize_frame(frame)

        except Exception as exc:
            last_error = exc

            print(
                f"Attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < retries:
                delay = retry_sleep * attempt
                print(f"Retrying after {delay:.1f} seconds...")
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to fetch {start_text} through {end_text}"
    ) from last_error


def manifest_record(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frame: pd.DataFrame,
    status: str,
) -> dict[str, Any]:
    missing_expected = sorted(
        EXPECTED_COLUMNS - set(frame.columns)
    )

    game_date_min = None
    game_date_max = None

    if "game_date" in frame.columns:
        valid_dates = pd.to_datetime(
            frame["game_date"],
            errors="coerce",
        ).dropna()

        if not valid_dates.empty:
            game_date_min = str(valid_dates.min().date())
            game_date_max = str(valid_dates.max().date())

    return {
        "chunk_start": start.strftime("%Y-%m-%d"),
        "chunk_end": end.strftime("%Y-%m-%d"),
        "file": str(path),
        "status": status,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "games": (
            int(frame["game_pk"].nunique())
            if "game_pk" in frame.columns
            else None
        ),
        "pitchers": (
            int(frame["pitcher"].nunique())
            if "pitcher" in frame.columns
            else None
        ),
        "batters": (
            int(frame["batter"].nunique())
            if "batter" in frame.columns
            else None
        ),
        "game_date_min": game_date_min,
        "game_date_max": game_date_max,
        "missing_expected_columns": "|".join(
            missing_expected
        ),
    }


def update_manifest(
    manifest_path: Path,
    record: dict[str, Any],
) -> pd.DataFrame:
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame()

    if not manifest.empty:
        keep = ~(
            manifest["chunk_start"].astype(str).eq(
                str(record["chunk_start"])
            )
            & manifest["chunk_end"].astype(str).eq(
                str(record["chunk_end"])
            )
        )

        manifest = manifest.loc[keep].copy()

    manifest = pd.concat(
        [manifest, pd.DataFrame([record])],
        ignore_index=True,
    )

    manifest = manifest.sort_values(
        ["chunk_start", "chunk_end"],
        kind="mergesort",
    )

    manifest.to_csv(manifest_path, index=False)

    return manifest


def main() -> None:
    args = parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    if end < start:
        raise ValueError("--end must be on or after --start")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    summary_path = out_dir / "fetch_summary.json"

    print("Enabling pybaseball cache...")
    cache.enable()

    chunks = monthly_chunks(start, end)

    completed = 0
    skipped = 0
    fetched_rows = 0

    for index, (chunk_start, chunk_end) in enumerate(
        chunks,
        start=1,
    ):
        filename = chunk_filename(chunk_start, chunk_end)
        output_path = out_dir / filename

        print()
        print("=" * 100)
        print(
            f"Chunk {index}/{len(chunks)}: "
            f"{chunk_start:%Y-%m-%d} through "
            f"{chunk_end:%Y-%m-%d}"
        )
        print(f"Output: {output_path}")

        readable, existing_rows = parquet_is_readable(
            output_path
        )

        should_skip = (
            args.resume
            and not args.overwrite
            and readable
        )

        if should_skip:
            print(
                "Completed chunk already exists; "
                f"skipping ({existing_rows:,} rows)."
            )

            existing = pd.read_parquet(output_path)

            record = manifest_record(
                output_path,
                chunk_start,
                chunk_end,
                existing,
                status="skipped_existing",
            )

            update_manifest(manifest_path, record)
            skipped += 1
            continue

        frame = fetch_chunk(
            start=chunk_start,
            end=chunk_end,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            parallel=args.parallel,
        )

        missing_expected = sorted(
            EXPECTED_COLUMNS - set(frame.columns)
        )

        if len(frame) and missing_expected:
            print(
                "WARNING — expected columns missing: "
                f"{missing_expected}"
            )

        temporary_path = output_path.with_suffix(
            ".parquet.tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        frame.to_parquet(
            temporary_path,
            index=False,
        )

        temporary_path.replace(output_path)

        readable, saved_rows = parquet_is_readable(
            output_path
        )

        if not readable:
            raise RuntimeError(
                f"Saved Parquet failed validation: {output_path}"
            )

        print(
            f"Saved {saved_rows:,} rows to {output_path}"
        )

        record = manifest_record(
            output_path,
            chunk_start,
            chunk_end,
            frame,
            status="downloaded",
        )

        update_manifest(manifest_path, record)

        completed += 1
        fetched_rows += int(len(frame))

    manifest = pd.read_csv(manifest_path)

    successful = manifest[
        manifest["status"].isin(
            ["downloaded", "skipped_existing"]
        )
    ].copy()

    summary = {
        "safety_version": SAFETY_VERSION,
        "requested_start": str(start.date()),
        "requested_end": str(end.date()),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "chunks_requested": int(len(chunks)),
        "chunks_downloaded_this_run": int(completed),
        "chunks_skipped_this_run": int(skipped),
        "rows_downloaded_this_run": int(fetched_rows),
        "manifest_rows": int(len(manifest)),
        "successful_manifest_rows": int(len(successful)),
        "total_rows_in_successful_chunks": int(
            pd.to_numeric(
                successful["rows"],
                errors="coerce",
            ).fillna(0).sum()
        ),
        "note": (
            "Raw pitch-level Statcast chunks only. "
            "No model features have been created yet."
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    print()
    print("=" * 100)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
