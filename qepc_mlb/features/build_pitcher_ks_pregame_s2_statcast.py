#!/usr/bin/env python3
"""
Build QEPC-MLB S2 prior-only pitch-level Statcast features.

Safety contract:
- Only regular-season Statcast pitches are used.
- A target start may use pitches only when:
      source_game_date < target_game_date
- Same-date games never feed one another.
- Target-game pitches cannot enter target-row features.
- Statcast pitcher IDs are mapped to the historical pitcher IDs in S0.
- Only explicitly registered S2 features enter modeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SAFETY_VERSION = "pitcher_ks_pregame_s2_statcast_v1"

WINDOWS = [100, 250, 500, 1000]

REQUIRED_PITCH_COLUMNS = [
    "game_date",
    "game_pk",
    "game_type",
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
]

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul",
    "foul_tip",
    "foul_bunt",
    "bunt_foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

STRIKEOUT_EVENTS = {
    "strikeout",
    "strikeout_double_play",
}

FASTBALL_TYPES = {
    "FF",
    "SI",
    "FC",
    "FA",
}

BREAKING_TYPES = {
    "SL",
    "ST",
    "SV",
    "CU",
    "KC",
    "CS",
}

OFFSPEED_TYPES = {
    "CH",
    "FS",
    "FO",
    "SC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s0",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s0.parquet"
        ),
    )
    parser.add_argument(
        "--s0_registry",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s0.feature_registry.json"
        ),
    )
    parser.add_argument(
        "--statcast_dir",
        default="cache/mlb/raw/statcast_pitch_level",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "cache/mlb/raw/statcast_pitch_level/"
            "manifest.csv"
        ),
    )
    parser.add_argument(
        "--crosswalk_source",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_enriched_opp_env_lineup.parquet"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "cache/mlb/processed/"
            "pitcher_game_logs_pregame_s2_statcast.parquet"
        ),
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

    raise ValueError(f"Unsupported file type: {path}")


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full(
        len(numerator),
        np.nan,
        dtype=float,
    )

    valid = denominator > 0

    result[valid] = (
        numerator[valid] / denominator[valid]
    )

    return result


def detect_crosswalk_id_column(
    columns: list[str],
) -> str:
    candidates = [
        "mlb_pitcher_id",
        "player_id",
        "pitcher_mlbam",
        "mlbam_id",
        "key_mlbam",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        "Could not locate an MLBAM pitcher ID column in "
        "the crosswalk source."
    )


def load_pitcher_crosswalk(
    path: Path,
) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    schema_columns = pq.ParquetFile(path).schema.names

    if "pitcher_id" not in schema_columns:
        raise ValueError(
            "Crosswalk source is missing pitcher_id."
        )

    mlb_id_column = detect_crosswalk_id_column(
        schema_columns
    )

    frame = pd.read_parquet(
        path,
        columns=["pitcher_id", mlb_id_column],
    )

    frame["pitcher_id"] = (
        frame["pitcher_id"]
        .astype("string")
        .str.strip()
    )

    frame["mlb_pitcher_id"] = pd.to_numeric(
        frame[mlb_id_column],
        errors="coerce",
    ).astype("Int64")

    frame = frame.dropna(
        subset=["pitcher_id", "mlb_pitcher_id"]
    ).copy()

    # Pick the most frequently observed mapping when duplicate
    # historical rows exist.
    counts = (
        frame.groupby(
            ["pitcher_id", "mlb_pitcher_id"],
            as_index=False,
        )
        .size()
        .sort_values(
            ["pitcher_id", "size"],
            ascending=[True, False],
            kind="mergesort",
        )
    )

    best = counts.drop_duplicates(
        "pitcher_id",
        keep="first",
    )[
        ["pitcher_id", "mlb_pitcher_id"]
    ].copy()

    return best, mlb_id_column


def load_statcast_chunks(
    manifest_path: Path,
    no_progress: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)

    manifest = manifest[
        manifest["status"].isin(
            ["downloaded", "skipped_existing"]
        )
    ].copy()

    records = manifest.to_dict(orient="records")

    iterator = records

    if tqdm is not None and not no_progress:
        iterator = tqdm(
            records,
            desc="Reading Statcast chunks",
            unit="chunk",
        )

    pieces: list[pd.DataFrame] = []

    rows_read = 0
    regular_rows = 0
    nonregular_rows = 0
    empty_chunks = 0

    for record in iterator:
        path = Path(record["file"])

        if not path.exists():
            raise FileNotFoundError(path)

        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)

        if parquet.metadata.num_rows == 0:
            empty_chunks += 1
            continue

        missing = sorted(
            set(REQUIRED_PITCH_COLUMNS) - available
        )

        if missing:
            raise RuntimeError(
                f"{path} is missing required columns: "
                f"{missing}"
            )

        frame = pd.read_parquet(
            path,
            columns=REQUIRED_PITCH_COLUMNS,
        )

        rows_read += int(len(frame))

        game_type = (
            frame["game_type"]
            .astype("string")
            .str.upper()
        )

        regular_mask = game_type.eq("R")

        regular_rows += int(regular_mask.sum())
        nonregular_rows += int((~regular_mask).sum())

        frame = frame.loc[regular_mask].copy()

        if not frame.empty:
            pieces.append(frame)

    if not pieces:
        raise RuntimeError(
            "No regular-season Statcast rows were loaded."
        )

    pitches = pd.concat(
        pieces,
        ignore_index=True,
    )

    metadata = {
        "manifest_rows": int(len(manifest)),
        "empty_chunks": int(empty_chunks),
        "rows_read_from_nonempty_chunks": int(rows_read),
        "regular_season_rows": int(regular_rows),
        "nonregular_rows_dropped": int(nonregular_rows),
    }

    return pitches, metadata


def prepare_pitch_flags(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    frame = pitches.copy()

    frame["game_date"] = pd.to_datetime(
        frame["game_date"],
        errors="coerce",
    ).dt.normalize()

    frame["mlb_pitcher_id"] = pd.to_numeric(
        frame["pitcher"],
        errors="coerce",
    ).astype("Int64")

    frame["game_pk"] = pd.to_numeric(
        frame["game_pk"],
        errors="coerce",
    )

    frame["at_bat_number"] = pd.to_numeric(
        frame["at_bat_number"],
        errors="coerce",
    )

    frame["pitch_number"] = pd.to_numeric(
        frame["pitch_number"],
        errors="coerce",
    )

    frame["zone"] = pd.to_numeric(
        frame["zone"],
        errors="coerce",
    )

    frame["strikes"] = pd.to_numeric(
        frame["strikes"],
        errors="coerce",
    )

    frame["release_speed"] = pd.to_numeric(
        frame["release_speed"],
        errors="coerce",
    )

    frame["pitch_type"] = (
        frame["pitch_type"]
        .astype("string")
        .str.upper()
        .fillna("")
    )

    description = (
        frame["description"]
        .astype("string")
        .str.lower()
        .fillna("")
    )

    events = (
        frame["events"]
        .astype("string")
        .str.lower()
        .fillna("")
    )

    pitch_result_type = (
        frame["type"]
        .astype("string")
        .str.upper()
        .fillna("")
    )

    frame = frame.dropna(
        subset=[
            "game_date",
            "mlb_pitcher_id",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
    ).copy()

    pitch_key = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "mlb_pitcher_id",
        "batter",
    ]

    frame = frame.drop_duplicates(
        pitch_key,
        keep="last",
    )

    frame = frame.sort_values(
        [
            "mlb_pitcher_id",
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    frame["flag_pitch"] = 1.0

    frame["flag_whiff"] = (
        description.isin(WHIFF_DESCRIPTIONS)
    ).astype(float)

    frame["flag_called_strike"] = (
        description.eq("called_strike")
    ).astype(float)

    frame["flag_csw"] = (
        frame["flag_whiff"]
        + frame["flag_called_strike"]
    ).clip(upper=1.0)

    frame["flag_swing"] = (
        description.isin(SWING_DESCRIPTIONS)
    ).astype(float)

    frame["flag_contact"] = (
        (frame["flag_swing"] == 1)
        & (frame["flag_whiff"] == 0)
    ).astype(float)

    frame["flag_zone_known"] = (
        frame["zone"].notna()
    ).astype(float)

    frame["flag_in_zone"] = (
        frame["zone"].between(1, 9)
    ).astype(float)

    frame["flag_out_zone"] = (
        (frame["flag_zone_known"] == 1)
        & (frame["flag_in_zone"] == 0)
    ).astype(float)

    frame["flag_chase"] = (
        (frame["flag_swing"] == 1)
        & (frame["flag_out_zone"] == 1)
    ).astype(float)

    frame["flag_zone_swing"] = (
        (frame["flag_swing"] == 1)
        & (frame["flag_in_zone"] == 1)
    ).astype(float)

    frame["flag_zone_contact"] = (
        (frame["flag_contact"] == 1)
        & (frame["flag_in_zone"] == 1)
    ).astype(float)

    frame["flag_first_pitch"] = (
        frame["pitch_number"] == 1
    ).astype(float)

    frame["flag_first_pitch_strike"] = (
        (frame["pitch_number"] == 1)
        & pitch_result_type.isin(["S", "X"])
    ).astype(float)

    frame["flag_two_strike_pa"] = (
        (frame["strikes"] == 2)
        & events.ne("")
    ).astype(float)

    frame["flag_strikeout_event"] = (
        events.isin(STRIKEOUT_EVENTS)
    ).astype(float)

    frame["flag_fastball"] = (
        frame["pitch_type"].isin(FASTBALL_TYPES)
    ).astype(float)

    frame["flag_breaking"] = (
        frame["pitch_type"].isin(BREAKING_TYPES)
    ).astype(float)

    frame["flag_offspeed"] = (
        frame["pitch_type"].isin(OFFSPEED_TYPES)
    ).astype(float)

    frame["flag_classified"] = (
        frame[
            [
                "flag_fastball",
                "flag_breaking",
                "flag_offspeed",
            ]
        ].sum(axis=1) > 0
    ).astype(float)

    for family in [
        "fastball",
        "breaking",
        "offspeed",
    ]:
        frame[f"flag_{family}_swing"] = (
            (frame[f"flag_{family}"] == 1)
            & (frame["flag_swing"] == 1)
        ).astype(float)

        frame[f"flag_{family}_whiff"] = (
            (frame[f"flag_{family}"] == 1)
            & (frame["flag_whiff"] == 1)
        ).astype(float)

    fastball_velocity_valid = (
        (frame["flag_fastball"] == 1)
        & frame["release_speed"].notna()
    )

    frame["fastball_velocity_sum"] = np.where(
        fastball_velocity_valid,
        frame["release_speed"],
        0.0,
    )

    frame["fastball_velocity_count"] = (
        fastball_velocity_valid.astype(float)
    )

    return frame


SUM_COLUMNS = [
    "flag_pitch",
    "flag_whiff",
    "flag_called_strike",
    "flag_csw",
    "flag_swing",
    "flag_contact",
    "flag_zone_known",
    "flag_in_zone",
    "flag_out_zone",
    "flag_chase",
    "flag_zone_swing",
    "flag_zone_contact",
    "flag_first_pitch",
    "flag_first_pitch_strike",
    "flag_two_strike_pa",
    "flag_strikeout_event",
    "flag_fastball",
    "flag_breaking",
    "flag_offspeed",
    "flag_classified",
    "flag_fastball_swing",
    "flag_fastball_whiff",
    "flag_breaking_swing",
    "flag_breaking_whiff",
    "flag_offspeed_swing",
    "flag_offspeed_whiff",
    "fastball_velocity_sum",
    "fastball_velocity_count",
]


def build_pitcher_features(
    targets: pd.DataFrame,
    pitches: pd.DataFrame,
    no_progress: bool,
) -> tuple[pd.DataFrame, list[str]]:
    output = pd.DataFrame(
        index=targets.index
    )

    output["s2_statcast_context_available"] = 0.0
    output["s2_statcast_prior_pitch_count"] = 0.0
    output["s2_latest_source_date"] = pd.NaT

    feature_names = [
        "s2_statcast_context_available",
        "s2_statcast_prior_pitch_count",
    ]

    for window in WINDOWS:
        names = [
            f"s2_pitch_count_{window}",
            f"s2_pitch_coverage_{window}",
            f"s2_swstr_rate_{window}",
            f"s2_csw_rate_{window}",
            f"s2_zone_rate_{window}",
            f"s2_chase_rate_{window}",
            f"s2_zone_contact_rate_{window}",
            f"s2_first_pitch_strike_rate_{window}",
            f"s2_two_strike_putaway_rate_{window}",
            f"s2_fastball_velocity_{window}",
            f"s2_fastball_mix_{window}",
            f"s2_breaking_mix_{window}",
            f"s2_offspeed_mix_{window}",
            f"s2_fastball_whiff_per_swing_{window}",
            f"s2_breaking_whiff_per_swing_{window}",
            f"s2_offspeed_whiff_per_swing_{window}",
            f"s2_pitch_mix_entropy_{window}",
        ]

        for name in names:
            output[name] = np.nan

        feature_names.extend(names)

    target_groups = (
        targets.dropna(subset=["mlb_pitcher_id"])
        .groupby("mlb_pitcher_id")
    )

    pitch_groups = {
        int(pitcher_id): group.reset_index(drop=True)
        for pitcher_id, group in pitches.groupby(
            "mlb_pitcher_id",
            sort=False,
        )
    }

    group_items = list(target_groups)

    iterator = group_items

    if tqdm is not None and not no_progress:
        iterator = tqdm(
            group_items,
            desc="Building prior-only Statcast features",
            unit="pitcher",
        )

    for pitcher_id, target_group in iterator:
        pitcher_key = int(pitcher_id)

        if pitcher_key not in pitch_groups:
            continue

        pitch_group = pitch_groups[pitcher_key]

        source_dates = (
            pitch_group["game_date"]
            .to_numpy(dtype="datetime64[D]")
        )

        target_indices = target_group.index.to_numpy()

        target_dates = (
            targets.loc[
                target_indices,
                "game_date",
            ]
            .to_numpy(dtype="datetime64[D]")
        )

        # side="left" means pitches on the target date are excluded.
        ends = np.searchsorted(
            source_dates,
            target_dates,
            side="left",
        )

        output.loc[
            target_indices,
            "s2_statcast_prior_pitch_count",
        ] = ends.astype(float)

        output.loc[
            target_indices,
            "s2_statcast_context_available",
        ] = (ends > 0).astype(float)

        latest_dates = np.full(
            len(ends),
            np.datetime64("NaT"),
            dtype="datetime64[D]",
        )

        has_prior = ends > 0

        latest_dates[has_prior] = source_dates[
            ends[has_prior] - 1
        ]

        output.loc[
            target_indices,
            "s2_latest_source_date",
        ] = pd.to_datetime(latest_dates)

        cumulative = {}

        for column in SUM_COLUMNS:
            values = pd.to_numeric(
                pitch_group[column],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            cumulative[column] = np.concatenate(
                [
                    np.array([0.0]),
                    np.cumsum(values),
                ]
            )

        for window in WINDOWS:
            starts = np.maximum(0, ends - window)
            counts = ends - starts

            sums = {
                column: (
                    cumulative[column][ends]
                    - cumulative[column][starts]
                )
                for column in SUM_COLUMNS
            }

            output.loc[
                target_indices,
                f"s2_pitch_count_{window}",
            ] = counts.astype(float)

            output.loc[
                target_indices,
                f"s2_pitch_coverage_{window}",
            ] = np.clip(
                counts.astype(float) / float(window),
                0.0,
                1.0,
            )

            output.loc[
                target_indices,
                f"s2_swstr_rate_{window}",
            ] = safe_divide(
                sums["flag_whiff"],
                sums["flag_pitch"],
            )

            output.loc[
                target_indices,
                f"s2_csw_rate_{window}",
            ] = safe_divide(
                sums["flag_csw"],
                sums["flag_pitch"],
            )

            output.loc[
                target_indices,
                f"s2_zone_rate_{window}",
            ] = safe_divide(
                sums["flag_in_zone"],
                sums["flag_zone_known"],
            )

            output.loc[
                target_indices,
                f"s2_chase_rate_{window}",
            ] = safe_divide(
                sums["flag_chase"],
                sums["flag_out_zone"],
            )

            output.loc[
                target_indices,
                f"s2_zone_contact_rate_{window}",
            ] = safe_divide(
                sums["flag_zone_contact"],
                sums["flag_zone_swing"],
            )

            output.loc[
                target_indices,
                f"s2_first_pitch_strike_rate_{window}",
            ] = safe_divide(
                sums["flag_first_pitch_strike"],
                sums["flag_first_pitch"],
            )

            output.loc[
                target_indices,
                f"s2_two_strike_putaway_rate_{window}",
            ] = safe_divide(
                sums["flag_strikeout_event"],
                sums["flag_two_strike_pa"],
            )

            output.loc[
                target_indices,
                f"s2_fastball_velocity_{window}",
            ] = safe_divide(
                sums["fastball_velocity_sum"],
                sums["fastball_velocity_count"],
            )

            fastball_mix = safe_divide(
                sums["flag_fastball"],
                sums["flag_classified"],
            )

            breaking_mix = safe_divide(
                sums["flag_breaking"],
                sums["flag_classified"],
            )

            offspeed_mix = safe_divide(
                sums["flag_offspeed"],
                sums["flag_classified"],
            )

            output.loc[
                target_indices,
                f"s2_fastball_mix_{window}",
            ] = fastball_mix

            output.loc[
                target_indices,
                f"s2_breaking_mix_{window}",
            ] = breaking_mix

            output.loc[
                target_indices,
                f"s2_offspeed_mix_{window}",
            ] = offspeed_mix

            output.loc[
                target_indices,
                f"s2_fastball_whiff_per_swing_{window}",
            ] = safe_divide(
                sums["flag_fastball_whiff"],
                sums["flag_fastball_swing"],
            )

            output.loc[
                target_indices,
                f"s2_breaking_whiff_per_swing_{window}",
            ] = safe_divide(
                sums["flag_breaking_whiff"],
                sums["flag_breaking_swing"],
            )

            output.loc[
                target_indices,
                f"s2_offspeed_whiff_per_swing_{window}",
            ] = safe_divide(
                sums["flag_offspeed_whiff"],
                sums["flag_offspeed_swing"],
            )

            mix_matrix = np.column_stack(
                [
                    fastball_mix,
                    breaking_mix,
                    offspeed_mix,
                ]
            )

            entropy_terms = np.where(
                mix_matrix > 0,
                mix_matrix * np.log(mix_matrix),
                0.0,
            )

            entropy = -np.nansum(
                entropy_terms,
                axis=1,
            )

            all_missing = np.isnan(
                mix_matrix
            ).all(axis=1)

            entropy[all_missing] = np.nan

            output.loc[
                target_indices,
                f"s2_pitch_mix_entropy_{window}",
            ] = entropy

    delta_specs = [
        "swstr_rate",
        "csw_rate",
        "zone_rate",
        "chase_rate",
        "zone_contact_rate",
        "first_pitch_strike_rate",
        "two_strike_putaway_rate",
        "fastball_velocity",
        "fastball_mix",
        "breaking_mix",
        "offspeed_mix",
        "pitch_mix_entropy",
    ]

    for metric in delta_specs:
        short_column = f"s2_{metric}_100"
        long_column = f"s2_{metric}_500"
        delta_column = f"s2_{metric}_delta_100_500"

        output[delta_column] = (
            output[short_column]
            - output[long_column]
        )

        feature_names.append(delta_column)

    return output, feature_names


def main() -> None:
    args = parse_args()

    s0_path = Path(args.s0)
    registry_path = Path(args.s0_registry)
    manifest_path = Path(args.manifest)
    crosswalk_path = Path(args.crosswalk_source)
    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading S0 dataset: {s0_path}")
    s0 = read_table(s0_path)

    s0["game_date"] = pd.to_datetime(
        s0["game_date"],
        errors="coerce",
    ).dt.normalize()

    s0["pitcher_id"] = (
        s0["pitcher_id"]
        .astype("string")
        .str.strip()
    )

    print(f"Loading pitcher ID mapping: {crosswalk_path}")
    crosswalk, source_id_column = (
        load_pitcher_crosswalk(crosswalk_path)
    )

    s0 = s0.merge(
        crosswalk,
        how="left",
        on="pitcher_id",
        validate="many_to_one",
    )

    id_match_count = int(
        s0["mlb_pitcher_id"].notna().sum()
    )

    print("Loading regular-season Statcast chunks...")
    pitches_raw, load_meta = load_statcast_chunks(
        manifest_path=manifest_path,
        no_progress=args.no_progress,
    )

    print("Preparing pitch-level flags...")
    pitches = prepare_pitch_flags(pitches_raw)

    source_date_min = str(
        pitches["game_date"].min().date()
    )
    source_date_max = str(
        pitches["game_date"].max().date()
    )

    print("Building target-row Statcast features...")
    statcast_features, s2_numeric_features = (
        build_pitcher_features(
            targets=s0,
            pitches=pitches,
            no_progress=args.no_progress,
        )
    )

    output = pd.concat(
        [
            s0.reset_index(drop=True),
            statcast_features.reset_index(drop=True),
        ],
        axis=1,
    )

    latest_source = pd.to_datetime(
        output["s2_latest_source_date"],
        errors="coerce",
    )

    target_date = pd.to_datetime(
        output["game_date"],
        errors="coerce",
    )

    violations = (
        latest_source.notna()
        & target_date.notna()
        & (latest_source >= target_date)
    )

    violation_count = int(violations.sum())

    if violation_count:
        raise RuntimeError(
            "Pregame safety violation: "
            f"{violation_count} rows used same-day or future "
            "Statcast pitches."
        )

    s0_registry = json.loads(
        registry_path.read_text()
    )

    numeric_features = list(
        s0_registry.get("numeric_features", [])
    )

    categorical_features = list(
        s0_registry.get("categorical_features", [])
    )

    for feature in s2_numeric_features:
        if feature not in numeric_features:
            numeric_features.append(feature)

    feature_metadata = {
        feature: {
            "source": "pitch_level_statcast",
            "time_rule": (
                "source_game_date < target_game_date"
            ),
            "pregame_safe": True,
        }
        for feature in s2_numeric_features
    }

    registry = {
        "safety_version": SAFETY_VERSION,
        "parent_registry": str(registry_path),
        "prediction_contract": (
            "All S2 Statcast features use regular-season "
            "pitches from dates strictly before the target date."
        ),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "s2_numeric_features": s2_numeric_features,
        "feature_metadata": feature_metadata,
        "forbidden_features": s0_registry.get(
            "forbidden_features",
            [],
        ),
    }

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

    output_registry_path = output_path.with_suffix(
        ".feature_registry.json"
    )

    output_summary_path = output_path.with_suffix(
        ".summary.json"
    )

    output_registry_path.write_text(
        json.dumps(
            registry,
            indent=2,
            default=str,
        )
    )

    available_count = int(
        (
            output[
                "s2_statcast_context_available"
            ] == 1
        ).sum()
    )

    at_least_100_count = int(
        (
            pd.to_numeric(
                output["s2_statcast_prior_pitch_count"],
                errors="coerce",
            ).fillna(0) >= 100
        ).sum()
    )

    starter_mask = (
        output["is_starter"]
        .astype("string")
        .str.lower()
        .isin(["true", "1", "yes", "starter", "sp"])
    )

    starter_rows = output.loc[starter_mask]

    summary = {
        "safety_version": SAFETY_VERSION,
        "s0_input": str(s0_path),
        "statcast_manifest": str(manifest_path),
        "crosswalk_source": str(crosswalk_path),
        "crosswalk_mlb_id_column": source_id_column,
        "out": str(output_path),
        "csv": (
            str(csv_path)
            if csv_path is not None
            else None
        ),
        "feature_registry": str(output_registry_path),
        "rows_written": int(len(output)),
        "columns_written": int(len(output.columns)),
        "source_date_min_regular_season": source_date_min,
        "source_date_max_regular_season": source_date_max,
        **load_meta,
        "pitch_rows_after_deduplication": int(
            len(pitches)
        ),
        "pitchers_in_statcast": int(
            pitches["mlb_pitcher_id"].nunique()
        ),
        "pitcher_id_match_count": id_match_count,
        "pitcher_id_match_rate": float(
            id_match_count / len(output)
        )
        if len(output)
        else None,
        "rows_with_any_prior_statcast": available_count,
        "rows_with_any_prior_statcast_rate": float(
            available_count / len(output)
        )
        if len(output)
        else None,
        "rows_with_at_least_100_prior_pitches": (
            at_least_100_count
        ),
        "starter_rows": int(len(starter_rows)),
        "starter_rows_with_any_prior_statcast": int(
            (
                starter_rows[
                    "s2_statcast_context_available"
                ] == 1
            ).sum()
        ),
        "s2_numeric_feature_count": int(
            len(s2_numeric_features)
        ),
        "total_numeric_feature_count": int(
            len(numeric_features)
        ),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "same_or_future_source_date_violations": (
            violation_count
        ),
        "note": (
            "S2 adds strictly prior-date pitch-level Statcast "
            "skill, velocity, pitch-mix, and whiff features."
        ),
    }

    output_summary_path.write_text(
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
