#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


S0_REGISTRY = Path(
    "cache/mlb/processed/"
    "pitcher_game_logs_pregame_s0.feature_registry.json"
)

S2_REGISTRY = Path(
    "cache/mlb/processed/"
    "pitcher_game_logs_pregame_s2_statcast.feature_registry.json"
)

OUT_DIR = Path(
    "cache/mlb/processed/s2_ablation_registries"
)


DIRECT_SKILL_METRICS = [
    "swstr_rate",
    "csw_rate",
    "zone_rate",
    "chase_rate",
    "zone_contact_rate",
    "first_pitch_strike_rate",
    "two_strike_putaway_rate",
]

VELOCITY_METRICS = [
    "fastball_velocity",
]

FAMILY_WHIFF_METRICS = [
    "fastball_whiff_per_swing",
    "breaking_whiff_per_swing",
    "offspeed_whiff_per_swing",
]

PITCH_MIX_METRICS = [
    "fastball_mix",
    "breaking_mix",
    "offspeed_mix",
    "pitch_mix_entropy",
]


def unique(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


def is_context_feature(feature: str) -> bool:
    return feature in {
        "s2_statcast_context_available",
        "s2_statcast_prior_pitch_count",
    }


def is_window_support(
    feature: str,
    windows: list[int],
) -> bool:
    return any(
        feature == f"s2_pitch_count_{window}"
        or feature == f"s2_pitch_coverage_{window}"
        for window in windows
    )


def contains_metric(
    feature: str,
    metrics: list[str],
) -> bool:
    return any(
        f"s2_{metric}_" in feature
        for metric in metrics
    )


def belongs_to_windows(
    feature: str,
    windows: list[int],
) -> bool:
    return any(
        feature.endswith(f"_{window}")
        for window in windows
    )


def select_direct_skill(
    features: list[str],
    windows: list[int],
    include_deltas: bool,
) -> list[str]:
    selected = []

    for feature in features:
        if is_context_feature(feature):
            selected.append(feature)
            continue

        if is_window_support(feature, windows):
            selected.append(feature)
            continue

        if not contains_metric(
            feature,
            DIRECT_SKILL_METRICS,
        ):
            continue

        if belongs_to_windows(feature, windows):
            selected.append(feature)
            continue

        if (
            include_deltas
            and feature.endswith("_delta_100_500")
        ):
            selected.append(feature)

    return unique(selected)


def add_family(
    base: list[str],
    features: list[str],
    metrics: list[str],
) -> list[str]:
    selected = list(base)

    for feature in features:
        if contains_metric(feature, metrics):
            selected.append(feature)

    return unique(selected)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s0 = json.loads(S0_REGISTRY.read_text())
    s2 = json.loads(S2_REGISTRY.read_text())

    s0_numeric = list(
        s0.get("numeric_features", [])
    )

    categorical = list(
        s0.get("categorical_features", [])
    )

    s2_features = list(
        s2.get("s2_numeric_features", [])
    )

    stable_core = select_direct_skill(
        s2_features,
        windows=[500, 1000],
        include_deltas=False,
    )

    recent_core = select_direct_skill(
        s2_features,
        windows=[100, 250],
        include_deltas=True,
    )

    core_skill = select_direct_skill(
        s2_features,
        windows=[100, 250, 500, 1000],
        include_deltas=True,
    )

    groups = {
        "stable_core": stable_core,
        "recent_core": recent_core,
        "core_skill": core_skill,
        "core_plus_velocity": add_family(
            core_skill,
            s2_features,
            VELOCITY_METRICS,
        ),
        "core_plus_family_whiff": add_family(
            core_skill,
            s2_features,
            FAMILY_WHIFF_METRICS,
        ),
        "core_plus_pitch_mix": add_family(
            core_skill,
            s2_features,
            PITCH_MIX_METRICS,
        ),
    }

    summaries = {}

    for name, selected_s2 in groups.items():
        payload = {
            "safety_version": (
                f"pitcher_ks_s2_ablation_{name}_v1"
            ),
            "parent_s0_registry": str(S0_REGISTRY),
            "parent_s2_registry": str(S2_REGISTRY),
            "ablation_name": name,
            "numeric_features": unique(
                s0_numeric + selected_s2
            ),
            "categorical_features": categorical,
            "s2_numeric_features": selected_s2,
            "s2_feature_count": len(selected_s2),
            "forbidden_features": s0.get(
                "forbidden_features",
                [],
            ),
            "prediction_contract": (
                "All S2 features use regular-season "
                "pitches from dates strictly before "
                "the target date."
            ),
        }

        path = OUT_DIR / f"{name}.json"

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

        summaries[name] = {
            "registry": str(path),
            "s2_feature_count": len(selected_s2),
            "total_numeric_count": len(
                payload["numeric_features"]
            ),
            "s2_features": selected_s2,
        }

    summary_path = OUT_DIR / "summary.json"

    summary_path.write_text(
        json.dumps(
            summaries,
            indent=2,
        )
    )

    print(
        json.dumps(
            summaries,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
