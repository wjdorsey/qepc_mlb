#!/usr/bin/env python3
"""
Fetch MLB batter actuals from MLB Stats API boxscores.

Output:
  cache/mlb/results/mlb_batter_actuals_YYYYMMDD.parquet
  cache/mlb/results/mlb_batter_actuals_YYYYMMDD.csv

This gives the results checker a web/API source instead of requiring local
Retrosheet-style batter logs to already be updated.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SAFETY_VERSION = "fetch_mlb_batter_actuals_v1_statsapi_boxscore"


MLB_TO_RETRO_TEAM = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CHW": "CHA",
    "CWS": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "OAK": "ATH",
    "ATH": "ATH",
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
    "WSN": "WAS",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default=None)
    p.add_argument("--also_csv", action="store_true", default=True)
    return p.parse_args()


def fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "qepc-mlb/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def statsapi_url(path: str, params: Dict[str, Any] | None = None) -> str:
    base = f"https://statsapi.mlb.com/api/v1/{path.lstrip('/')}"
    if params:
        return base + "?" + urllib.parse.urlencode(params)
    return base


def normalize_name(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace(".", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b[a-z]\b", " ", s)  # drop middle initials like Michael A Taylor
    return re.sub(r"\s+", " ", s).strip()


def team_code(team_obj: Dict[str, Any]) -> str:
    raw = (
        team_obj.get("abbreviation")
        or team_obj.get("fileCode")
        or team_obj.get("teamCode")
        or team_obj.get("name")
        or ""
    )
    raw = str(raw).upper()
    return MLB_TO_RETRO_TEAM.get(raw, raw)


def as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def parse_boxscore_game(game_pk: int, game_date: str) -> List[Dict[str, Any]]:
    url = statsapi_url(f"game/{game_pk}/boxscore")
    box = fetch_json(url)

    rows: List[Dict[str, Any]] = []

    teams = box.get("teams", {})
    away_team = teams.get("away", {}).get("team", {})
    home_team = teams.get("home", {}).get("team", {})

    away_code = team_code(away_team)
    home_code = team_code(home_team)

    for side, opponent_side in [("away", "home"), ("home", "away")]:
        side_data = teams.get(side, {})
        opp_data = teams.get(opponent_side, {})

        team = away_code if side == "away" else home_code
        opponent = home_code if side == "away" else away_code

        players = side_data.get("players", {})
        batter_ids = side_data.get("batters", []) or []

        batting_order = side_data.get("battingOrder", []) or []
        batting_order_rank = {
            str(pid).replace("ID", ""): i + 1
            for i, pid in enumerate(batting_order)
        }

        for pid in batter_ids:
            pid_str = str(pid).replace("ID", "")
            player = players.get(f"ID{pid_str}", {})
            person = player.get("person", {})
            stats = player.get("stats", {}).get("batting", {}) or {}

            # Some listed batters may have zero PA; keep them, evaluator can still decide.
            hits = as_int(stats.get("hits"))
            at_bats = as_int(stats.get("atBats"))
            pa = as_int(stats.get("plateAppearances"))

            batter_name = person.get("fullName") or player.get("name") or ""

            rows.append(
                {
                    "game_pk": str(game_pk),
                    "game_id": str(game_pk),
                    "game_date": game_date,
                    "team": team,
                    "opponent": opponent,
                    "mlb_batter_id": pid_str,
                    "batter_name": batter_name,
                    "batter_name_key": normalize_name(batter_name),
                    "batting_order": batting_order_rank.get(pid_str),
                    "plate_appearances": pa,
                    "at_bats": at_bats,
                    "hits": hits,
                    "hit_1plus": int(hits >= 1),
                    "runs": as_int(stats.get("runs")),
                    "rbi": as_int(stats.get("rbi")),
                    "walks": as_int(stats.get("baseOnBalls")),
                    "strikeouts": as_int(stats.get("strikeOuts")),
                    "total_bases": as_int(stats.get("totalBases")),
                    "home_runs": as_int(stats.get("homeRuns")),
                    "doubles": as_int(stats.get("doubles")),
                    "triples": as_int(stats.get("triples")),
                }
            )

    return rows


def main() -> None:
    args = parse_args()

    date_tag = args.date.replace("-", "")
    out_path = Path(args.out) if args.out else Path(f"cache/mlb/results/mlb_batter_actuals_{date_tag}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schedule_url = statsapi_url("schedule", {"sportId": 1, "date": args.date})
    schedule = fetch_json(schedule_url)

    games = []
    for d in schedule.get("dates", []):
        games.extend(d.get("games", []))

    rows: List[Dict[str, Any]] = []
    game_summaries = []

    for g in games:
        game_pk = g.get("gamePk")
        status = (g.get("status") or {}).get("detailedState")
        if not game_pk:
            continue

        game_rows = parse_boxscore_game(int(game_pk), args.date)
        rows.extend(game_rows)

        game_summaries.append(
            {
                "game_pk": str(game_pk),
                "status": status,
                "away": team_code((g.get("teams", {}).get("away", {}).get("team", {}))),
                "home": team_code((g.get("teams", {}).get("home", {}).get("team", {}))),
                "rows": len(game_rows),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["game_pk", "mlb_batter_id", "team"], keep="first")

    df.to_parquet(out_path, index=False)

    csv_path = out_path.with_suffix(".csv")
    if args.also_csv:
        df.to_csv(csv_path, index=False)

    summary = {
        "safety_version": SAFETY_VERSION,
        "date": args.date,
        "schedule_games": int(len(games)),
        "rows": int(len(df)),
        "teams": sorted(df["team"].dropna().unique().tolist()) if not df.empty else [],
        "out": str(out_path),
        "csv": str(csv_path) if args.also_csv else None,
        "game_summaries": game_summaries,
        "note": "Fetched batter actuals from MLB Stats API boxscores.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
