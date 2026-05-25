#!/usr/bin/env python3
"""
QEPC-MLB Batter 1+ Hit Scoring Slate Builder.

Purpose:
  Converts a manually entered confirmed-lineup CSV into a prediction-ready slate
  for the Batter 1+ Hit Ranker model.

Workflow:
  1. Create/fill a lineup CSV.
  2. Build scoreable slate with this script.
  3. Score it with predict_batter_1plus_hit_ranker.py.

Safety:
  - Uses only historical batter rows before game_date.
  - Uses only historical pitcher rows before game_date.
  - Does not use same-game outcomes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SAFETY_VERSION = "batter_1plus_hit_scoring_slate_v1_pregame_safe"


HITTER_STATS = [
    "hit_1plus",
    "hits",
    "at_bats",
    "plate_appearances",
    "total_bases",
    "walks",
    "strikeouts",
    "home_runs",
]

PITCHER_STATS = [
    "strikeouts",
    "batters_faced",
    "hits_allowed",
    "walks_allowed",
    "runs_allowed",
    "earned_runs",
    "home_runs_allowed",
    "outs_recorded",
    "innings_pitched",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Batter 1+ Hit prediction-ready scoring slate")

    p.add_argument("--write_template", default=None, help="Write a blank manual lineup CSV template and exit")

    p.add_argument("--lineup_csv", default=None, help="Manual lineup CSV")
    p.add_argument("--history", default="cache/mlb/processed/batter_game_logs_opp_pitcher.parquet")
    p.add_argument("--pitchers", default="cache/mlb/processed/pitcher_game_logs.parquet")
    p.add_argument("--model_dir", default="artifacts/mlb/models/batter_1plus_hit_ranker_champion_v2")
    p.add_argument("--out", default=None)
    p.add_argument("--also_csv", action="store_true")

    return p.parse_args()


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "game_date",
        "game_id",
        "team",
        "opponent",
        "home_away",
        "batter_id",
        "batter_name",
        "batter_hand",
        "lineup_slot",
        "opp_starter_id",
        "opp_starter_name",
        "opp_starter_throw",
        "park_id",
        "env_temp",
        "env_windspeed",
        "env_winddir",
        "env_daynight",
        "env_usedh",
    ]

    example = {
        "game_date": "2026-04-01",
        "game_id": "20260401_CHN_SLN",
        "team": "CHN",
        "opponent": "SLN",
        "home_away": "home",
        "batter_id": "hoern001",
        "batter_name": "Nico Hoerner",
        "batter_hand": "R",
        "lineup_slot": 2,
        "opp_starter_id": "pitch001",
        "opp_starter_name": "Opposing Starter",
        "opp_starter_throw": "R",
        "park_id": "CHC11",
        "env_temp": 68,
        "env_windspeed": 8,
        "env_winddir": "out",
        "env_daynight": "day",
        "env_usedh": True,
    }

    pd.DataFrame([example], columns=cols).to_csv(path, index=False)
    print(f"Wrote template: {path}")


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def normalize_name(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    elif "date" in df.columns:
        raw = df["date"].astype("string").str.replace(r"\.0$", "", regex=True)
        df["game_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("Missing game_date/date column")

    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def coalesce_numeric(df: pd.DataFrame, names: Sequence[str], default=np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype="float64")
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            out = out.where(out.notna(), s)
    return out


def normalize_pitchers(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p = ensure_date(p)

    if "stattype" in p.columns:
        p = p[p["stattype"].astype("string").str.lower().eq("value")].copy()

    pitcher_col = first_existing(p, ["pitcher_id", "id"])
    game_col = first_existing(p, ["game_id", "gid"])
    team_col = first_existing(p, ["team", "pitcher_team"])

    if pitcher_col is None:
        raise ValueError("Pitcher file missing pitcher_id/id column")

    out = pd.DataFrame(index=p.index)
    out["pitcher_id"] = p[pitcher_col].astype("string")
    out["game_id"] = p[game_col].astype("string") if game_col else pd.NA
    out["pitcher_team"] = p[team_col].astype("string") if team_col else pd.NA
    out["game_date"] = p["game_date"]
    out["season"] = p["season"]

    if "is_starter" in p.columns:
        out["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(float).ne(0)
    elif "p_gs" in p.columns:
        out["is_starter"] = pd.to_numeric(p["p_gs"], errors="coerce").fillna(0).astype(float).ne(0)
    else:
        out["is_starter"] = False

    out["strikeouts"] = coalesce_numeric(p, ["strikeouts", "p_k"])
    out["batters_faced"] = coalesce_numeric(p, ["batters_faced", "p_bfp"])
    out["hits_allowed"] = coalesce_numeric(p, ["hits_allowed", "p_h"])
    out["walks_allowed"] = coalesce_numeric(p, ["walks_allowed", "walks", "p_w"])
    out["runs_allowed"] = coalesce_numeric(p, ["runs_allowed", "p_r"])
    out["earned_runs"] = coalesce_numeric(p, ["earned_runs", "p_er"])
    out["home_runs_allowed"] = coalesce_numeric(p, ["home_runs_allowed", "p_hr"])
    out["outs_recorded"] = coalesce_numeric(p, ["outs_recorded", "p_ipouts"])

    if "innings_pitched" in p.columns:
        out["innings_pitched"] = pd.to_numeric(p["innings_pitched"], errors="coerce")
    else:
        out["innings_pitched"] = out["outs_recorded"] / 3.0

    out = out.dropna(subset=["pitcher_id", "game_date"]).copy()
    out = out[out["is_starter"]].copy()
    return out


def latest_batter_id_by_name(history: pd.DataFrame) -> Dict[str, str]:
    if "batter_name" not in history.columns:
        return {}

    id_col = first_existing(history, ["batter_id", "id"])
    if id_col is None:
        return {}

    h = history.dropna(subset=["batter_name", id_col, "game_date"]).copy()
    h["_name_key"] = h["batter_name"].map(normalize_name)
    h = h.sort_values("game_date").drop_duplicates("_name_key", keep="last")
    return dict(zip(h["_name_key"], h[id_col].astype("string")))


def get_lineup_value(row: pd.Series, names: Sequence[str], default=None):
    for n in names:
        if n in row.index and pd.notna(row[n]) and str(row[n]).strip() != "":
            return row[n]
    return default


def add_hitter_features(row: pd.Series, history: pd.DataFrame, batter_id: str, game_date: pd.Timestamp) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    id_col = first_existing(history, ["batter_id", "id"])
    if id_col is None:
        raise ValueError("History missing batter_id/id column")

    h = history[(history[id_col].astype("string") == str(batter_id)) & (history["game_date"] < game_date)].copy()
    h = h.sort_values("game_date")

    out["batter_games_prior"] = float(len(h))
    out["games_prior"] = float(len(h))

    if len(h):
        out["batter_days_rest"] = float((game_date - h["game_date"].iloc[-1]).days)
    else:
        out["batter_days_rest"] = np.nan

    for stat in HITTER_STATS:
        if stat not in h.columns:
            for suffix in ["lag1", "roll3", "roll5", "roll10", "roll20"]:
                out[f"{stat}_{suffix}"] = np.nan
            continue

        s = pd.to_numeric(h[stat], errors="coerce")
        out[f"{stat}_lag1"] = float(s.iloc[-1]) if len(s) else np.nan

        for w in [3, 5, 10, 20]:
            out[f"{stat}_roll{w}"] = float(s.tail(w).mean()) if len(s) else np.nan

    for w in [5, 10, 20]:
        recent = h.tail(w)
        hits = pd.to_numeric(recent.get("hits", pd.Series(dtype=float)), errors="coerce").sum()
        ab = pd.to_numeric(recent.get("at_bats", pd.Series(dtype=float)), errors="coerce").sum()
        pa = pd.to_numeric(recent.get("plate_appearances", pd.Series(dtype=float)), errors="coerce").sum()
        ks = pd.to_numeric(recent.get("strikeouts", pd.Series(dtype=float)), errors="coerce").sum()
        bb = pd.to_numeric(recent.get("walks", pd.Series(dtype=float)), errors="coerce").sum()

        out[f"batter_ba_roll{w}"] = float(hits / ab) if ab else np.nan
        out[f"batter_k_rate_roll{w}"] = float(ks / pa) if pa else np.nan
        out[f"batter_bb_rate_roll{w}"] = float(bb / pa) if pa else np.nan

    season = int(game_date.year)
    hs = h[h["season"].astype(float) == season].copy()

    hits = pd.to_numeric(hs.get("hits", pd.Series(dtype=float)), errors="coerce").sum()
    ab = pd.to_numeric(hs.get("at_bats", pd.Series(dtype=float)), errors="coerce").sum()
    pa = pd.to_numeric(hs.get("plate_appearances", pd.Series(dtype=float)), errors="coerce").sum()
    ks = pd.to_numeric(hs.get("strikeouts", pd.Series(dtype=float)), errors="coerce").sum()
    bb = pd.to_numeric(hs.get("walks", pd.Series(dtype=float)), errors="coerce").sum()

    out["hits_season_prior"] = float(hits)
    out["at_bats_season_prior"] = float(ab)
    out["plate_appearances_season_prior"] = float(pa)
    out["strikeouts_season_prior"] = float(ks)
    out["walks_season_prior"] = float(bb)
    out["batter_ba_season_prior"] = float(hits / ab) if ab else np.nan
    out["batter_k_rate_season_prior"] = float(ks / pa) if pa else np.nan
    out["batter_bb_rate_season_prior"] = float(bb / pa) if pa else np.nan

    # Fallback identity/context from latest history row if lineup CSV omitted them.
    if len(h):
        last = h.iloc[-1]
        for c in ["batter_name", "batter_hand"]:
            if c in h.columns and c not in out:
                out[c] = last.get(c)

    return out


def add_pitcher_features(
    pitcher_history: pd.DataFrame,
    pitcher_id: str,
    game_date: pd.Timestamp,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    h = pitcher_history[
        (pitcher_history["pitcher_id"].astype("string") == str(pitcher_id))
        & (pitcher_history["game_date"] < game_date)
    ].copy()
    h = h.sort_values("game_date")

    out["has_opp_starter_context"] = bool(len(h))
    out["opp_starter_games_prior"] = float(len(h))

    if len(h):
        out["opp_starter_days_rest"] = float((game_date - h["game_date"].iloc[-1]).days)
    else:
        out["opp_starter_days_rest"] = np.nan

    for stat in PITCHER_STATS:
        if stat not in h.columns:
            for suffix in ["lag1", "roll3", "roll5", "roll10"]:
                out[f"opp_starter_{stat}_{suffix}"] = np.nan
            continue

        s = pd.to_numeric(h[stat], errors="coerce")
        out[f"opp_starter_{stat}_lag1"] = float(s.iloc[-1]) if len(s) else np.nan

        for w in [3, 5, 10]:
            out[f"opp_starter_{stat}_roll{w}"] = float(s.tail(w).mean()) if len(s) else np.nan

    rate_specs = {
        "k_per_bf": ("strikeouts", "batters_faced"),
        "hits_per_bf": ("hits_allowed", "batters_faced"),
        "bb_per_bf": ("walks_allowed", "batters_faced"),
        "hr_per_bf": ("home_runs_allowed", "batters_faced"),
        "er_per_ip": ("earned_runs", "innings_pitched"),
    }

    for rate_name, (num_col, den_col) in rate_specs.items():
        if num_col not in h.columns or den_col not in h.columns or h.empty:
            out[f"opp_starter_{rate_name}_lag1"] = np.nan
            for w in [3, 5, 10]:
                out[f"opp_starter_{rate_name}_roll{w}"] = np.nan
            continue

        num = pd.to_numeric(h[num_col], errors="coerce")
        den = pd.to_numeric(h[den_col], errors="coerce").replace(0, np.nan)

        out[f"opp_starter_{rate_name}_lag1"] = float(num.iloc[-1] / den.iloc[-1]) if pd.notna(den.iloc[-1]) else np.nan

        for w in [3, 5, 10]:
            nsum = num.tail(w).sum()
            dsum = den.tail(w).sum()
            out[f"opp_starter_{rate_name}_roll{w}"] = float(nsum / dsum) if dsum else np.nan

    return out



def normalize_hand(x: Any) -> str:
    if pd.isna(x):
        return "U"

    s = str(x).strip().upper()

    if s in {"R", "RIGHT", "RIGHTY"}:
        return "R"
    if s in {"L", "LEFT", "LEFTY"}:
        return "L"
    if s in {"S", "B", "SW", "SWITCH", "BOTH"}:
        return "S"

    return "U"


def add_matchup_features_to_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    batter_hand = normalize_hand(rec.get("batter_hand"))
    pitcher_throw = normalize_hand(rec.get("opp_starter_throw"))

    rec["batter_hand_norm"] = batter_hand
    rec["opp_starter_throw_norm"] = pitcher_throw

    rec["batter_bats_right"] = int(batter_hand == "R")
    rec["batter_bats_left"] = int(batter_hand == "L")
    rec["batter_bats_switch"] = int(batter_hand == "S")

    rec["opp_starter_throws_right"] = int(pitcher_throw == "R")
    rec["opp_starter_throws_left"] = int(pitcher_throw == "L")

    known_batter = batter_hand in {"R", "L", "S"}
    known_pitcher = pitcher_throw in {"R", "L"}
    known_matchup = known_batter and known_pitcher

    same_hand = (
        (batter_hand == "R" and pitcher_throw == "R")
        or (batter_hand == "L" and pitcher_throw == "L")
    )

    opposite_hand = (
        (batter_hand == "R" and pitcher_throw == "L")
        or (batter_hand == "L" and pitcher_throw == "R")
    )

    platoon_advantage = opposite_hand or (batter_hand == "S" and known_pitcher)

    rec["matchup_known"] = int(known_matchup)
    rec["same_hand_matchup"] = int(same_hand) if known_matchup else np.nan
    rec["opposite_hand_matchup"] = int(opposite_hand) if known_matchup else np.nan
    rec["platoon_advantage"] = int(platoon_advantage) if known_matchup else np.nan
    rec["batter_pitcher_matchup_code"] = f"{batter_hand}_vs_{pitcher_throw}"

    return rec


def main() -> None:
    args = parse_args()

    if args.write_template:
        write_template(Path(args.write_template))
        return

    if not args.lineup_csv:
        raise ValueError("--lineup_csv is required unless --write_template is used")

    lineup = pd.read_csv(args.lineup_csv)
    history = ensure_date(read_any(args.history))
    pitchers = normalize_pitchers(read_any(args.pitchers))

    model_dir = Path(args.model_dir)
    feature_list = load_json(model_dir / "feature_list.json")
    expected_numeric = feature_list.get("numeric_features", [])
    expected_categorical = feature_list.get("categorical_features", [])

    batter_name_map = latest_batter_id_by_name(history)

    rows: List[Dict[str, Any]] = []
    missing_batter_ids = 0
    missing_pitcher_ids = 0

    for _, row in lineup.iterrows():
        game_date = pd.to_datetime(row["game_date"], errors="coerce")
        if pd.isna(game_date):
            raise ValueError(f"Bad game_date in lineup row: {row.to_dict()}")

        batter_id = get_lineup_value(row, ["batter_id", "id"], default=None)
        batter_name = get_lineup_value(row, ["batter_name", "name", "player_name"], default=None)

        if batter_id is None and batter_name is not None:
            batter_id = batter_name_map.get(normalize_name(batter_name))

        if batter_id is None:
            missing_batter_ids += 1
            continue

        opp_starter_id = get_lineup_value(row, ["opp_starter_id", "opposing_starter_id"], default=None)
        if opp_starter_id is None:
            missing_pitcher_ids += 1

        rec: Dict[str, Any] = {}
        rec["game_date"] = game_date
        rec["season"] = int(game_date.year)
        rec["game_id"] = get_lineup_value(
            row,
            ["game_id", "gid"],
            default=f"{game_date.strftime('%Y%m%d')}_{row.get('team', 'UNK')}_{row.get('opponent', 'UNK')}",
        )

        rec["batter_id"] = str(batter_id)
        rec["id"] = str(batter_id)
        rec["batter_name"] = batter_name
        rec["team"] = get_lineup_value(row, ["team"], default=None)
        rec["opponent"] = get_lineup_value(row, ["opponent", "opp"], default=None)
        rec["home_away"] = str(get_lineup_value(row, ["home_away", "vishome"], default="unknown")).lower()
        rec["is_home"] = rec["home_away"] in {"home", "h", "1", "true"}
        rec["lineup_slot"] = pd.to_numeric(get_lineup_value(row, ["lineup_slot", "b_lp"], default=np.nan), errors="coerce")
        rec["is_starting_lineup"] = bool(pd.notna(rec["lineup_slot"]) and 1 <= float(rec["lineup_slot"]) <= 9)
        rec["is_starting_lineup_by_batting_file"] = rec["is_starting_lineup"]
        rec["is_starting_lineup_teamstats"] = rec["is_starting_lineup"]
        rec["is_starter"] = rec["is_starting_lineup"]

        rec["batter_hand"] = get_lineup_value(row, ["batter_hand", "bat"], default=None)

        rec["park_id"] = get_lineup_value(row, ["park_id", "site"], default=None)
        rec["env_temp"] = pd.to_numeric(get_lineup_value(row, ["env_temp", "temp"], default=np.nan), errors="coerce")
        rec["env_windspeed"] = pd.to_numeric(get_lineup_value(row, ["env_windspeed", "windspeed"], default=np.nan), errors="coerce")
        rec["env_winddir"] = get_lineup_value(row, ["env_winddir", "winddir"], default="unknown")
        rec["env_daynight"] = get_lineup_value(row, ["env_daynight", "daynight"], default="unknown")
        rec["env_usedh"] = get_lineup_value(row, ["env_usedh", "usedh"], default="unknown")
        rec["has_env_context"] = pd.notna(rec["env_temp"]) or pd.notna(rec["env_windspeed"])

        rec["opp_starter_id"] = str(opp_starter_id) if opp_starter_id is not None else pd.NA
        rec["opp_starter_name"] = get_lineup_value(row, ["opp_starter_name", "opposing_starter_name"], default=None)
        rec["opp_starter_throw"] = get_lineup_value(row, ["opp_starter_throw", "opposing_starter_throw"], default=None)

        rec.update(add_hitter_features(row, history, str(batter_id), game_date))

        if opp_starter_id is not None:
            rec.update(add_pitcher_features(pitchers, str(opp_starter_id), game_date))
        else:
            rec["has_opp_starter_context"] = False

        rec = add_matchup_features_to_record(rec)
        rows.append(rec)

    slate = pd.DataFrame(rows)

    # Ensure every expected model feature exists.
    for c in expected_numeric:
        if c not in slate.columns:
            slate[c] = np.nan
    for c in expected_categorical:
        if c not in slate.columns:
            slate[c] = "__MISSING__"

    out_path = Path(args.out) if args.out else Path(
        f"cache/mlb/scoring/batter_1plus_hit_slate_{pd.to_datetime(slate['game_date'].max()).strftime('%Y%m%d')}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slate.to_parquet(out_path, index=False)

    csv_path = None
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        slate.to_csv(csv_path, index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "lineup_csv": args.lineup_csv,
        "history": args.history,
        "pitchers": args.pitchers,
        "model_dir": args.model_dir,
        "rows_in_lineup_csv": int(len(lineup)),
        "rows_written": int(len(slate)),
        "missing_batter_ids_skipped": int(missing_batter_ids),
        "missing_pitcher_ids": int(missing_pitcher_ids),
        "dates": [str(x.date()) for x in sorted(pd.to_datetime(slate["game_date"]).dropna().unique())],
        "expected_numeric_features": int(len(expected_numeric)),
        "expected_categorical_features": int(len(expected_categorical)),
        "missing_expected_numeric_after_build": [c for c in expected_numeric if c not in slate.columns],
        "missing_expected_categorical_after_build": [c for c in expected_categorical if c not in slate.columns],
        "has_opp_starter_context_rate": float(slate["has_opp_starter_context"].mean()) if "has_opp_starter_context" in slate.columns and len(slate) else None,
        "out": str(out_path),
        "csv": str(csv_path) if csv_path else None,
        "note": "This slate is prediction-ready for predict_batter_1plus_hit_ranker.py.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
