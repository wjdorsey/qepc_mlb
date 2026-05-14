from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from qepc_mlb.config import PATHS  # type: ignore
except Exception:  # pragma: no cover
    PATHS = None


KEY_COLS = ["player_id", "season"]


def iter_progress(items, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def load_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")



def first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        found = lower_map.get(cand.lower())
        if found is not None:
            return found
    return None



def empty_string_series(index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")



def parse_name(frame: pd.DataFrame) -> pd.Series:
    col = first_existing(frame.columns, ["player_name", "last_name, first_name", "name"])
    if col is None:
        return empty_string_series(frame.index)
    return frame[col].astype("string").str.strip()



def get_numeric(frame: pd.DataFrame, colname: Optional[str]) -> pd.Series:
    if colname is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[colname], errors="coerce")



def base_table(frame: pd.DataFrame, name_col: str) -> pd.DataFrame:
    cols = frame.columns.tolist()
    out = pd.DataFrame(index=frame.index)
    out["player_id"] = get_numeric(frame, first_existing(cols, ["player_id", "playerid", "pitcher_id", "mlbam_id"]))
    out["season"] = get_numeric(frame, first_existing(cols, ["year", "season"]))
    out[name_col] = parse_name(frame)
    return out



def normalize_expected(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = base_table(frame, "pitcher_name_expected")
    mapping = {
        "pa": "savant_pa",
        "bip": "savant_bip",
        "ba": "savant_ba_allowed",
        "est_ba": "savant_xba_allowed",
        "est_ba_minus_ba_diff": "savant_xba_minus_ba",
        "slg": "savant_slg_allowed",
        "est_slg": "savant_xslg_allowed",
        "est_slg_minus_slg_diff": "savant_xslg_minus_slg",
        "woba": "savant_woba_allowed",
        "est_woba": "savant_xwoba_allowed",
        "est_woba_minus_woba_diff": "savant_xwoba_minus_woba",
        "era": "savant_era",
        "xera": "savant_xera",
        "era_minus_xera_diff": "savant_era_minus_xera",
    }
    for src, dst in mapping.items():
        if src in frame.columns:
            out[dst] = pd.to_numeric(frame[src], errors="coerce")
    out["source_expected"] = source_name
    return out



def normalize_percentile(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = base_table(frame, "pitcher_name_percentile")
    mapping = {
        "xwoba": "savant_pct_xwoba",
        "xba": "savant_pct_xba",
        "xslg": "savant_pct_xslg",
        "xiso": "savant_pct_xiso",
        "xobp": "savant_pct_xobp",
        "brl": "savant_pct_brl",
        "brl_percent": "savant_pct_brl_rate",
        "exit_velocity": "savant_pct_exit_velocity",
        "max_ev": "savant_pct_max_ev",
        "hard_hit_percent": "savant_pct_hard_hit",
        "k_percent": "savant_pct_k",
        "bb_percent": "savant_pct_bb",
        "whiff_percent": "savant_pct_whiff",
        "chase_percent": "savant_pct_chase",
        "arm_strength": "savant_pct_arm_strength",
        "xera": "savant_pct_xera",
        "fb_velocity": "savant_pct_fb_velocity",
        "fb_spin": "savant_pct_fb_spin",
        "curve_spin": "savant_pct_curve_spin",
    }
    for src, dst in mapping.items():
        if src in frame.columns:
            out[dst] = pd.to_numeric(frame[src], errors="coerce")
    out["source_percentile"] = source_name
    return out



def normalize_run_value(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = base_table(frame, "pitcher_name_run_value")
    mapping = {
        "team_id": "savant_team_id",
        "pa": "savant_pa_rv",
        "pitches": "savant_season_pitches",
        "runs_all": "savant_runs_all",
        "runs_heart": "savant_runs_heart",
        "runs_shadow": "savant_runs_shadow",
        "runs_chase": "savant_runs_chase",
        "runs_waste": "savant_runs_waste",
    }
    for src, dst in mapping.items():
        if src in frame.columns:
            out[dst] = pd.to_numeric(frame[src], errors="coerce")
    out["source_run_value"] = source_name
    return out



def classify_and_normalize(path: Path) -> Tuple[str, pd.DataFrame]:
    df = load_any(path)
    lower_name = path.name.lower()
    if "expected" in lower_name:
        return "expected", normalize_expected(df, path.name)
    if "percentile" in lower_name:
        return "percentile", normalize_percentile(df, path.name)
    if "run_value" in lower_name or "runvalue" in lower_name:
        return "run_value", normalize_run_value(df, path.name)
    raise ValueError(f"Could not classify Savant file type from filename: {path.name}")



def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()



def collapse_same_type(kind: str, frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"No frames to collapse for kind={kind}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = dedupe_columns(combined)
    combined = combined.dropna(subset=["player_id", "season"], how="any")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce")
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce")
    combined = combined.sort_values(["season", "player_id"]).drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    return combined



def coalesce_name(frame: pd.DataFrame) -> pd.Series:
    name_cols = [c for c in frame.columns if c.startswith("pitcher_name")]
    if not name_cols:
        return empty_string_series(frame.index)

    out = empty_string_series(frame.index)
    for c in name_cols:
        col_obj = frame.loc[:, c]
        if isinstance(col_obj, pd.DataFrame):
            for subcol in col_obj.columns:
                out = out.fillna(col_obj[subcol].astype("string"))
        else:
            out = out.fillna(col_obj.astype("string"))
    return out



def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"savant_pct_k", "savant_pct_bb"}.issubset(out.columns):
        out["savant_pct_k_minus_bb"] = out["savant_pct_k"] - out["savant_pct_bb"]
    if {"savant_runs_shadow", "savant_runs_chase", "savant_runs_waste", "savant_runs_heart"}.issubset(out.columns):
        out["savant_runs_command_combo"] = out["savant_runs_shadow"].fillna(0) + out["savant_runs_chase"].fillna(0)
        out["savant_runs_bad_zone_combo"] = out["savant_runs_heart"].fillna(0) + out["savant_runs_waste"].fillna(0)
    if {"savant_season_pitches", "savant_pa_rv"}.issubset(out.columns):
        denom = out["savant_pa_rv"].replace(0, np.nan)
        out["savant_pitches_per_pa"] = out["savant_season_pitches"] / denom
    return out



def main() -> None:
    parser = argparse.ArgumentParser(description="Build QEPC MLB pitcher-season Savant feature table")
    parser.add_argument("--sources", nargs="+", required=True, help="Savant CSV/Parquet files to merge")
    parser.add_argument("--out", default="cache/mlb/processed/pitcher_savant_features.parquet", help="Output parquet path")
    parser.add_argument("--also_csv", action="store_true", help="Also save CSV next to parquet")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    if PATHS is not None:
        PATHS.ensure()

    grouped: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    progress_on = not args.no_progress

    for src in iter_progress(args.sources, enabled=progress_on, desc="Loading Savant sources"):
        kind, norm = classify_and_normalize(Path(src))
        grouped[kind].append(norm)

    if not grouped:
        raise ValueError("No Savant frames were loaded.")

    print("Collapsing same-type Savant tables...")
    collapsed: Dict[str, pd.DataFrame] = {}
    for kind in sorted(grouped):
        collapsed[kind] = collapse_same_type(kind, grouped[kind])

    print("Merging Savant feature tables by player_id + season...")
    merge_order = [k for k in ["expected", "percentile", "run_value"] if k in collapsed]
    merged = collapsed[merge_order[0]].copy()
    for kind in merge_order[1:]:
        merged = merged.merge(collapsed[kind], how="outer", on=KEY_COLS)
        merged = dedupe_columns(merged)

    merged["pitcher_name"] = coalesce_name(merged)
    name_cols = [c for c in merged.columns if c.startswith("pitcher_name_")]
    if name_cols:
        merged = merged.drop(columns=name_cols)

    for col in [c for c in merged.columns if c.startswith("source_")]:
        merged[col] = merged[col].astype("string")

    merged = add_derived_features(merged)
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")
    merged = merged.sort_values(["season", "player_id"]).drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    feature_columns = [
        c for c in merged.columns
        if c not in {"player_id", "season", "pitcher_name"} and not c.startswith("source_")
    ]
    summary = {
        "rows": int(len(merged)),
        "pitchers": int(merged["player_id"].nunique(dropna=True)),
        "seasons": sorted([int(x) for x in merged["season"].dropna().unique().tolist()]),
        "feature_columns": feature_columns,
        "non_null_counts": {c: int(merged[c].notna().sum()) for c in feature_columns},
    }

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved parquet: {out_path}")
    print(f"saved summary: {summary_path}")
    print(json.dumps({
        "rows": summary["rows"],
        "pitchers": summary["pitchers"],
        "seasons": summary["seasons"],
        "n_features": len(summary["feature_columns"]),
    }, indent=2))

    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        merged.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")


if __name__ == "__main__":
    main()
