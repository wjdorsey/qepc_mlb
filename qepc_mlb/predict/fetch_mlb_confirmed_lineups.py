#!/usr/bin/env python3
"""
Fetch MLB confirmed lineups / probable starters and write a QEPC-compatible lineup CSV.

Output CSV is intended for:
  qepc_mlb/predict/build_batter_1plus_hit_scoring_slate.py

Source:
  MLB Stats API schedule + boxscore endpoints.

Notes:
  - Confirmed batting orders usually appear only after lineups are posted.
  - Player IDs from MLB API are not the same as Retrosheet IDs.
  - This script tries to map MLB names/team/season to Retrosheet IDs using allplayers.csv.
  - If an ID cannot be mapped, batter_name / opp_starter_name are still written.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import date as dt_date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


SAFETY_VERSION = "fetch_mlb_confirmed_lineups_v1_statsapi"

BASE = "https://statsapi.mlb.com/api/v1"

MLB_TEAM_ID_TO_ABBR = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "ATH",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}


MLB_TO_RETRO_TEAM = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CWS": "CHA",
    "CHW": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "KCR": "KCA",
    "LAA": "ANA",
    "ANA": "ANA",
    "LAD": "LAN",
    "LA": "LAN",
    "MIA": "MIA",
    "FLA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "ATH",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SDP": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "SFG": "SFN",
    "STL": "SLN",
    "TB": "TBA",
    "TBR": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
    "WAS": "WAS",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch MLB confirmed lineups into QEPC lineup CSV")
    p.add_argument("--date", default=None, help="YYYY-MM-DD. Defaults to today.")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--allplayers", default="cache/mlb/raw/retrosheet/allplayers.csv")
    p.add_argument("--include_unconfirmed", action="store_true", help="Write games even if batting order is not posted")
    p.add_argument("--summary_json", default=None)
    return p.parse_args()


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE}/{path.lstrip('/')}"
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def normalize_name(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""

    s = str(x).strip()

    # Convert accents/diacritics to ASCII:
    # Pérez -> Perez, Báez -> Baez, Acuña -> Acuna, José -> Jose
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    s = s.lower()
    s = s.replace(".", " ")

    # Remove common suffixes.
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)

    # Normalize punctuation/spacing.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mlb_abbrev(team_blob: Optional[Dict[str, Any]]) -> Optional[str]:
    if not team_blob:
        return None

    for key in ["abbreviation", "teamCode", "fileCode"]:
        val = team_blob.get(key)
        if val:
            val = str(val).upper()
            # MLB fileCode can be lowercase/alternate; normalize common ones.
            aliases = {
                "CHW": "CWS",
                "KCR": "KC",
                "SDP": "SD",
                "SFG": "SF",
                "TBR": "TB",
                "WSN": "WSH",
                "AZ": "ARI",
            }
            return aliases.get(val, val)

    tid = team_blob.get("id")
    try:
        tid = int(tid)
    except Exception:
        tid = None

    if tid in MLB_TEAM_ID_TO_ABBR:
        return MLB_TEAM_ID_TO_ABBR[tid]

    return None


def retro_team(abbrev: Optional[str]) -> Optional[str]:
    if not abbrev:
        return None
    return MLB_TO_RETRO_TEAM.get(str(abbrev).upper(), str(abbrev).upper())


def load_retrosheet_name_map(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"WARNING: allplayers file not found: {path}", file=sys.stderr)
        return pd.DataFrame()

    ap = pd.read_csv(
        p,
        usecols=lambda c: c in {"id", "last", "first", "bat", "throw", "team", "season", "g", "g_p"},
    )

    ap["season"] = pd.to_numeric(ap["season"], errors="coerce")
    ap["team"] = ap["team"].astype("string")
    ap["player_name"] = (
        ap["first"].astype("string").fillna("").str.strip()
        + " "
        + ap["last"].astype("string").fillna("").str.strip()
    ).str.strip()
    ap["name_key"] = ap["player_name"].map(normalize_name)
    ap["id"] = ap["id"].astype("string")

    if "g_p" not in ap.columns:
        ap["g_p"] = 0
    if "g" not in ap.columns:
        ap["g"] = 0

    return ap


def match_retro_id(
    ap: pd.DataFrame,
    *,
    name: Optional[str],
    team: Optional[str],
    season: int,
    pitcher: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    if ap.empty or not name:
        return None, "no_map_or_name"

    key = normalize_name(name)
    if not key:
        return None, "empty_name"

    cand = ap[ap["name_key"].eq(key)].copy()
    if cand.empty:
        return None, "name_not_found"

    if pitcher and "g_p" in cand.columns:
        pitcher_cand = cand[pd.to_numeric(cand["g_p"], errors="coerce").fillna(0) > 0].copy()
        if not pitcher_cand.empty:
            cand = pitcher_cand

    # Best: same season + same team
    if team:
        exact = cand[(cand["season"].eq(season)) & (cand["team"].astype("string").eq(str(team)))].copy()
        if not exact.empty:
            exact = exact.sort_values(["g", "g_p"], ascending=False)
            return str(exact["id"].iloc[0]), "season_team_name"

    # Same season
    same_season = cand[cand["season"].eq(season)].copy()
    if not same_season.empty:
        same_season = same_season.sort_values(["g", "g_p"], ascending=False)
        return str(same_season["id"].iloc[0]), "season_name"

    # Latest season <= target, preferably same team
    prior = cand[cand["season"].le(season)].copy()
    if not prior.empty:
        if team:
            prior_team = prior[prior["team"].astype("string").eq(str(team))].copy()
            if not prior_team.empty:
                prior_team = prior_team.sort_values(["season", "g", "g_p"], ascending=False)
                return str(prior_team["id"].iloc[0]), "prior_team_name"

        prior = prior.sort_values(["season", "g", "g_p"], ascending=False)
        return str(prior["id"].iloc[0]), "prior_name"

    # Total fallback
    cand = cand.sort_values(["season", "g", "g_p"], ascending=False)
    return str(cand["id"].iloc[0]), "fallback_name"


def get_probable_pitcher(game: Dict[str, Any], side: str) -> Tuple[Optional[int], Optional[str]]:
    team_blob = game.get("teams", {}).get(side, {})
    pp = team_blob.get("probablePitcher") or {}
    return pp.get("id"), pp.get("fullName")


def extract_side_lineup(
    *,
    box: Dict[str, Any],
    game: Dict[str, Any],
    side: str,
    game_date: str,
    ap: pd.DataFrame,
) -> List[Dict[str, Any]]:
    side_blob = box.get("teams", {}).get(side, {})
    players = side_blob.get("players", {}) or {}
    batter_ids = side_blob.get("batters", []) or []

    team_info = game.get("teams", {}).get(side, {}).get("team", {}) or {}
    opp_side = "away" if side == "home" else "home"
    opp_info = game.get("teams", {}).get(opp_side, {}).get("team", {}) or {}

    box_team_info = side_blob.get("team", {}) or {}
    box_opp_info = box.get("teams", {}).get(opp_side, {}).get("team", {}) or {}

    mlb_team_abbr = mlb_abbrev(box_team_info) or mlb_abbrev(team_info)
    mlb_opp_abbr = mlb_abbrev(box_opp_info) or mlb_abbrev(opp_info)

    team = retro_team(mlb_team_abbr)
    opponent = retro_team(mlb_opp_abbr)
    home_away = "home" if side == "home" else "away"

    opp_pp_id, opp_pp_name = get_probable_pitcher(game, opp_side)

    season = int(str(game_date)[:4])
    opp_starter_id, opp_starter_match_type = match_retro_id(
        ap,
        name=opp_pp_name,
        team=opponent,
        season=season,
        pitcher=True,
    )

    rows = []
    for pid in batter_ids:
        key = f"ID{pid}"
        pdata = players.get(key, {}) or {}
        person = pdata.get("person", {}) or {}
        name = person.get("fullName")

        order_raw = pdata.get("battingOrder")
        if order_raw is None:
            continue

        try:
            order_int = int(order_raw)
        except Exception:
            continue

        # Starters are normally 100,200,...,900. Subs may be 101, 201, etc.
        if order_int < 100 or order_int > 900 or order_int % 100 != 0:
            continue

        lineup_slot = order_int // 100

        retro_batter_id, batter_match_type = match_retro_id(
            ap,
            name=name,
            team=team,
            season=season,
            pitcher=False,
        )

        bat_side = pdata.get("batSide", {}).get("code")
        pos_name = pdata.get("position", {}).get("abbreviation")

        rows.append(
            {
                "game_date": game_date,
                "game_id": str(game.get("gamePk")),
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "batter_id": retro_batter_id,
                "batter_name": name,
                "batter_hand": bat_side,
                "lineup_slot": lineup_slot,
                "opp_starter_id": opp_starter_id,
                "opp_starter_name": opp_pp_name,
                "opp_starter_throw": None,
                "park_id": None,
                "env_temp": None,
                "env_windspeed": None,
                "env_winddir": "unknown",
                "env_daynight": game.get("dayNight", "unknown"),
                "env_usedh": "unknown",
                "mlb_game_pk": game.get("gamePk"),
                "mlb_batter_id": pid,
                "mlb_opp_starter_id": opp_pp_id,
                "mlb_team_abbr": mlb_team_abbr,
                "mlb_opp_abbr": mlb_opp_abbr,
                "position": pos_name,
                "batter_id_match_type": batter_match_type,
                "opp_starter_id_match_type": opp_starter_match_type,
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    game_date = args.date or dt_date.today().isoformat()
    out = Path(args.out or f"cache/mlb/manual/mlb_confirmed_lineups_{game_date.replace('-', '')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    ap = load_retrosheet_name_map(args.allplayers)

    schedule = api_get(
        "schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
        },
    )

    games = []
    for d in schedule.get("dates", []):
        games.extend(d.get("games", []))

    all_rows: List[Dict[str, Any]] = []
    games_with_lineups = 0
    games_without_lineups = 0

    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue

        try:
            box = api_get(f"game/{game_pk}/boxscore")
        except Exception as e:
            print(f"WARNING: failed boxscore for gamePk={game_pk}: {e}", file=sys.stderr)
            continue

        game_rows = []
        for side in ["away", "home"]:
            side_rows = extract_side_lineup(
                box=box,
                game=game,
                side=side,
                game_date=game_date,
                ap=ap,
            )
            game_rows.extend(side_rows)

        if game_rows:
            games_with_lineups += 1
            all_rows.extend(game_rows)
        else:
            games_without_lineups += 1
            if args.include_unconfirmed:
                # Write placeholder team/game rows so user can see schedule was found.
                for side in ["away", "home"]:
                    team_info = game.get("teams", {}).get(side, {}).get("team", {}) or {}
                    opp_side = "away" if side == "home" else "home"
                    opp_info = game.get("teams", {}).get(opp_side, {}).get("team", {}) or {}
                    pp_id, pp_name = get_probable_pitcher(game, opp_side)
                    all_rows.append(
                        {
                            "game_date": game_date,
                            "game_id": str(game_pk),
                            "team": retro_team(mlb_abbrev(team_info)),
                            "opponent": retro_team(mlb_abbrev(opp_info)),
                            "home_away": "home" if side == "home" else "away",
                            "batter_id": None,
                            "batter_name": None,
                            "batter_hand": None,
                            "lineup_slot": None,
                            "opp_starter_id": None,
                            "opp_starter_name": pp_name,
                            "opp_starter_throw": None,
                            "park_id": None,
                            "env_temp": None,
                            "env_windspeed": None,
                            "env_winddir": "unknown",
                            "env_daynight": game.get("dayNight", "unknown"),
                            "env_usedh": "unknown",
                            "mlb_game_pk": game_pk,
                            "mlb_batter_id": None,
                            "mlb_opp_starter_id": pp_id,
                            "mlb_team_abbr": mlb_abbrev(team_info),
                            "mlb_opp_abbr": mlb_abbrev(opp_info),
                            "position": None,
                            "batter_id_match_type": "unconfirmed_no_batter",
                            "opp_starter_id_match_type": "unconfirmed",
                        }
                    )

    fieldnames = [
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
        "mlb_game_pk",
        "mlb_batter_id",
        "mlb_opp_starter_id",
        "mlb_team_abbr",
        "mlb_opp_abbr",
        "position",
        "batter_id_match_type",
        "opp_starter_id_match_type",
    ]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    batter_id_rate = None
    pitcher_id_rate = None
    if all_rows:
        batter_id_rate = sum(bool(r.get("batter_id")) for r in all_rows) / len(all_rows)
        pitcher_id_rate = sum(bool(r.get("opp_starter_id")) for r in all_rows) / len(all_rows)

    summary = {
        "safety_version": SAFETY_VERSION,
        "date": game_date,
        "games_found": len(games),
        "games_with_lineups": games_with_lineups,
        "games_without_lineups": games_without_lineups,
        "rows_written": len(all_rows),
        "batter_id_match_rate": batter_id_rate,
        "opp_starter_id_match_rate": pitcher_id_rate,
        "out": str(out),
        "note": "If rows_written is 0, lineups probably are not posted yet for this date.",
    }

    summary_path = Path(args.summary_json) if args.summary_json else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
