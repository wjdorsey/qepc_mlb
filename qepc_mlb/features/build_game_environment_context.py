
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join game environment context from Retrosheet gameinfo.csv onto pitcher game logs.")
    p.add_argument("--input", required=True, help="Path to pitcher game logs parquet/csv (ideally enriched + opponent context logs).")
    p.add_argument("--gameinfo", required=True, help="Path to Retrosheet gameinfo.csv")
    p.add_argument("--out", default="cache/mlb/processed/pitcher_game_logs_enriched_opp_env.parquet", help="Output parquet/csv path.")
    p.add_argument("--min_season", type=int, default=None, help="Optional minimum season filter after join.")
    p.add_argument("--also_csv", action="store_true", help="Also save CSV alongside parquet.")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")


def save_any(df: pd.DataFrame, path: Path, also_csv: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        print(f"saved csv: {path}")
        return
    df.to_parquet(path, index=False)
    print(f"saved parquet: {path}")
    if also_csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")


def coerce_bool_like(series: pd.Series, true_values: set[str], false_values: set[str]) -> pd.Series:
    txt = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[txt.isin(true_values)] = True
    out.loc[txt.isin(false_values)] = False
    return out


def normalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    # try to preserve game id
    if "gid" in out.columns and "game_id" not in out.columns:
        out["game_id"] = out["gid"]
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].astype("string").str.strip()
    for c in ["team", "opponent", "home_away", "vishome", "site"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
    if "is_home" not in out.columns:
        if "home_away" in out.columns:
            ha = out["home_away"].astype("string").str.strip().str.lower()
            out["is_home"] = ha.isin(["h", "home"])
        elif "vishome" in out.columns:
            vh = out["vishome"].astype("string").str.strip().str.lower()
            out["is_home"] = vh.isin(["h", "home"])
        else:
            out["is_home"] = pd.NA
    return out


def normalize_gameinfo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # standard retrosheet columns usually already named these
    rename_map = {
        "gid": "game_id",
        "site": "park_id",
        "date": "game_date_raw",
        "visteam": "visteam",
        "hometeam": "hometeam",
        "winddir": "winddir",
        "windspeed": "windspeed",
        "temp": "temp",
        "daynight": "daynight",
        "usedh": "usedh",
        "number": "game_number",
    }
    for old, new in rename_map.items():
        if old in out.columns and new != old:
            out = out.rename(columns={old: new})
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].astype("string").str.strip()
    if "game_date_raw" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date_raw"].astype("string"), format="%Y%m%d", errors="coerce")
    elif "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    else:
        out["game_date"] = pd.NaT
    out["season"] = out["game_date"].dt.year.astype("Int64")
    for c in ["visteam", "hometeam", "park_id", "winddir", "daynight", "usedh"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
    for c in ["windspeed", "temp", "attendance", "duration", "starttime"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # derived environment signals
    if "temp" in out.columns:
        out["env_temp"] = pd.to_numeric(out["temp"], errors="coerce")
        out["env_temp_sq"] = out["env_temp"] ** 2
        out["env_temp_missing"] = out["env_temp"].isna()
    else:
        out["env_temp"] = np.nan
        out["env_temp_sq"] = np.nan
        out["env_temp_missing"] = True

    if "windspeed" in out.columns:
        out["env_windspeed"] = pd.to_numeric(out["windspeed"], errors="coerce")
        out["env_windspeed_sq"] = out["env_windspeed"] ** 2
        out["env_windspeed_missing"] = out["env_windspeed"].isna()
    else:
        out["env_windspeed"] = np.nan
        out["env_windspeed_sq"] = np.nan
        out["env_windspeed_missing"] = True

    wind = out.get("winddir", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").str.strip().str.lower()
    out["env_winddir"] = wind
    out["env_wind_out"] = wind.str.contains("out", na=False)
    out["env_wind_in"] = wind.str.contains("in", na=False)
    out["env_wind_cross"] = wind.str.contains("left|right|cross", na=False)
    out["env_wind_calm"] = wind.isin(["none", "unknown", ""])
    if "env_windspeed" in out.columns:
        out["env_wind_out_effect"] = np.where(out["env_wind_out"], out["env_windspeed"], 0.0)
        out["env_wind_in_effect"] = np.where(out["env_wind_in"], out["env_windspeed"], 0.0)
        out["env_wind_cross_effect"] = np.where(out["env_wind_cross"], out["env_windspeed"], 0.0)

    dn = out.get("daynight", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").str.strip().str.lower()
    out["env_daynight"] = dn
    out["env_is_night"] = dn.isin(["night", "n"])
    out["env_is_day"] = dn.isin(["day", "d"])

    usedh = out.get("usedh", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").str.strip().str.lower()
    out["env_usedh"] = usedh
    out["env_has_dh"] = usedh.isin(["true", "t", "yes", "y", "1"])
    out["env_no_dh"] = usedh.isin(["false", "f", "no", "n", "0"])

    if "park_id" in out.columns:
        out["env_park_id"] = out["park_id"].astype("string").str.strip()
    else:
        out["env_park_id"] = pd.NA

    keep = [
        "game_id", "game_date", "season", "visteam", "hometeam", "env_park_id",
        "env_temp", "env_temp_sq", "env_temp_missing",
        "env_windspeed", "env_windspeed_sq", "env_windspeed_missing",
        "env_winddir", "env_wind_out", "env_wind_in", "env_wind_cross", "env_wind_calm",
        "env_wind_out_effect", "env_wind_in_effect", "env_wind_cross_effect",
        "env_daynight", "env_is_night", "env_is_day",
        "env_usedh", "env_has_dh", "env_no_dh",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()
    out = out.drop_duplicates(subset=[c for c in ["game_id"] if c in out.columns], keep="last")
    return out


def attach_environment(logs: pd.DataFrame, gi: pd.DataFrame) -> pd.DataFrame:
    out = logs.copy()
    matched_by = "none"

    # Preferred: exact game_id join
    if "game_id" in out.columns and "game_id" in gi.columns and out["game_id"].notna().any() and gi["game_id"].notna().any():
        temp = gi.drop_duplicates(subset=["game_id"], keep="last")
        out = out.merge(temp, how="left", on="game_id", suffixes=("", "_env"))
        matched_by = "game_id"
    else:
        # Fallback on season + date + team/opponent + home/away against visteam/hometeam
        gi2 = gi.copy()
        gi2["join_home_team"] = gi2.get("hometeam", pd.Series(pd.NA, index=gi2.index, dtype="string")).astype("string")
        gi2["join_away_team"] = gi2.get("visteam", pd.Series(pd.NA, index=gi2.index, dtype="string")).astype("string")
        out["join_home_team"] = np.where(out["is_home"].fillna(False), out["team"], out["opponent"])
        out["join_away_team"] = np.where(out["is_home"].fillna(False), out["opponent"], out["team"])
        key_cols = ["season", "game_date", "join_home_team", "join_away_team"]
        temp = gi2.drop_duplicates(subset=key_cols, keep="last")
        out = out.merge(temp.drop(columns=["game_id"], errors="ignore"), how="left", on=key_cols, suffixes=("", "_env"))
        matched_by = "date+teams"

    out["has_env_context"] = out["env_park_id"].notna()
    # useful interaction style columns
    if "env_temp" in out.columns and "env_is_night" in out.columns:
        out["env_temp_night_interaction"] = np.where(out["env_is_night"].fillna(False), out["env_temp"], 0.0)
    if "env_windspeed" in out.columns and "env_is_home" not in out.columns:
        out["env_is_home"] = out.get("is_home", pd.Series(pd.NA, index=out.index))
    return out, matched_by


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    gpath = Path(args.gameinfo)
    out_path = Path(args.out)

    print("Loading pitcher game logs...")
    logs = normalize_logs(load_any(inp))
    print("Loading game environment source...")
    gi = normalize_gameinfo(load_any(gpath))

    print("Joining game environment context...")
    joined, matched_by = attach_environment(logs, gi)

    if args.min_season is not None:
        joined = joined[joined["season"].fillna(-1).astype(int) >= args.min_season].copy()

    save_any(joined, out_path, also_csv=args.also_csv)

    env_cols = [c for c in joined.columns if c.startswith("env_")] + (["has_env_context"] if "has_env_context" in joined.columns else [])
    summary = {
        "rows": int(len(joined)),
        "seasons": sorted([int(x) for x in joined["season"].dropna().astype(int).unique().tolist()]) if "season" in joined.columns else [],
        "matched_by": matched_by,
        "env_match_rows": int(joined["has_env_context"].fillna(False).sum()) if "has_env_context" in joined.columns else 0,
        "env_match_rate": float(joined["has_env_context"].fillna(False).mean()) if "has_env_context" in joined.columns and len(joined) else 0.0,
        "env_columns": env_cols,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
