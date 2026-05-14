
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# -----------------------------
# Basic path support
# -----------------------------
try:
    from qepc_mlb.config import PATHS  # type: ignore
except Exception:  # pragma: no cover
    PATHS = None

TEAM_ALIASES = {
    "angels": "LAA", "los angeles angels": "LAA",
    "diamondbacks": "ARI", "arizona diamondbacks": "ARI",
    "braves": "ATL", "atlanta braves": "ATL",
    "orioles": "BAL", "baltimore orioles": "BAL",
    "red sox": "BOS", "boston red sox": "BOS",
    "cubs": "CHC", "chicago cubs": "CHC",
    "white sox": "CWS", "chicago white sox": "CWS",
    "reds": "CIN", "cincinnati reds": "CIN",
    "guardians": "CLE", "cleveland guardians": "CLE", "indians": "CLE", "cleveland indians": "CLE",
    "rockies": "COL", "colorado rockies": "COL",
    "tigers": "DET", "detroit tigers": "DET",
    "astros": "HOU", "houston astros": "HOU",
    "royals": "KC", "kansas city royals": "KC",
    "dodgers": "LAD", "los angeles dodgers": "LAD",
    "marlins": "MIA", "miami marlins": "MIA", "florida marlins": "FLA",
    "brewers": "MIL", "milwaukee brewers": "MIL",
    "twins": "MIN", "minnesota twins": "MIN",
    "mets": "NYM", "new york mets": "NYM",
    "yankees": "NYY", "new york yankees": "NYY",
    "athletics": "ATH", "a's": "ATH", "oakland athletics": "OAK", "sacramento athletics": "ATH",
    "phillies": "PHI", "philadelphia phillies": "PHI",
    "pirates": "PIT", "pittsburgh pirates": "PIT",
    "padres": "SD", "san diego padres": "SD",
    "giants": "SF", "san francisco giants": "SF",
    "mariners": "SEA", "seattle mariners": "SEA",
    "cardinals": "STL", "st louis cardinals": "STL", "st. louis cardinals": "STL",
    "rays": "TB", "tampa bay rays": "TB", "devil rays": "TBD", "tampa bay devil rays": "TBD",
    "rangers": "TEX", "texas rangers": "TEX",
    "blue jays": "TOR", "toronto blue jays": "TOR",
    "nationals": "WSH", "washington nationals": "WSH",
    # Retrosheet historic short codes often already fine
}

CANONICAL_ALIASES: Dict[str, List[str]] = {
    "game_id": ["game_id", "gid", "game_pk"],
    "pitcher_id": ["pitcher_id", "player_id", "mlbam_id", "pitcher_mlbam_id", "id"],
    "pitcher_name": ["pitcher_name", "player_name", "name", "last_name, first_name"],
    "game_date": ["game_date", "date", "game_dt"],
    "season": ["season", "year"],
    "team": ["team", "pitching_team", "player_team"],
    "opponent": ["opponent", "opp", "batting_team", "opponent_team"],
    "home_away": ["home_away", "vishome", "homeaway"],
    "is_home": ["is_home", "home_flag"],
    "is_starter": ["is_starter", "starter", "started", "p_gs"],
    "innings_pitched": ["innings_pitched", "ip"],
    "outs_recorded": ["outs_recorded", "outs", "recorded_outs", "p_ipouts"],
    "strikeouts": ["strikeouts", "pitcher_strikeouts", "so", "k", "ks", "SO", "p_k"],
    "walks": ["walks", "bb", "base_on_balls", "BB", "p_w"],
    "hits_allowed": ["hits_allowed", "hits", "h", "H", "p_h"],
    "earned_runs": ["earned_runs", "er", "ER", "p_er"],
    "runs_allowed": ["runs_allowed", "runs", "r", "R", "p_r"],
    "batters_faced": ["batters_faced", "bf", "BF", "p_bfp"],
    "pitches": ["pitches", "pitch_count", "pit", "Pitches"],
    "pitcher_hand": ["pitcher_hand", "throws", "p_throws", "hand", "throws_hand", "throw"],
    "days_rest": ["days_rest", "rest_days"],
    "decision": ["decision", "dec"],
    "source_stattype": ["stattype"],
}

NUMERIC_COLUMNS = {
    "season", "innings_pitched", "outs_recorded", "strikeouts", "walks", "hits_allowed",
    "earned_runs", "runs_allowed", "batters_faced", "pitches", "days_rest",
}


def iter_progress(items, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def first_existing(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for alias in aliases:
        found = lower_map.get(alias.lower())
        if found is not None:
            return found
    return None


def load_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def parse_game_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    text = series.astype("string").str.strip()
    dt = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return dt.fillna(fallback)


def normalize_team_name(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if len(text) in (2, 3, 4):
        return text.upper()
    return TEAM_ALIASES.get(text.lower().replace(".", ""), text.upper())


def normalize_home_away(raw: pd.Series) -> pd.Series:
    text = raw.astype("string").str.strip().str.lower()
    mapped = pd.Series(pd.NA, index=raw.index, dtype="string")
    mapped[text.isin(["home", "h", "1", "true", "t", "yes"])] = "home"
    mapped[text.isin(["away", "a", "0", "false", "f", "no", "v", "visitor"])] = "away"
    return mapped


def coerce_is_starter(series: pd.Series) -> pd.Series:
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    text = s.astype("string").str.strip().str.lower()
    out.loc[out.isna() & text.isin(["sp", "starter", "start", "true", "t", "yes", "y"])] = True
    out.loc[out.isna() & text.isin(["rp", "reliever", "false", "f", "no", "n"])] = False
    return out


def normalize_frame(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = pd.DataFrame(index=df.index)
    mapping: Dict[str, str] = {}

    for canon, aliases in CANONICAL_ALIASES.items():
        src_col = first_existing(df.columns, aliases)
        if src_col is not None:
            out[canon] = df[src_col]
            mapping[canon] = src_col
        else:
            out[canon] = pd.Series([pd.NA] * len(df), index=df.index)

    # additional name support from first/last in raw file
    if out["pitcher_name"].isna().all():
        first_name_col = first_existing(df.columns, ["first_name", "firstname", "name_first", "first"])
        last_name_col = first_existing(df.columns, ["last_name", "lastname", "name_last", "last"])
        if first_name_col or last_name_col:
            first = df[first_name_col].fillna("").astype(str).str.strip() if first_name_col else pd.Series("", index=df.index)
            last = df[last_name_col].fillna("").astype(str).str.strip() if last_name_col else pd.Series("", index=df.index)
            name = (pd.Series(first, index=df.index).astype(str) + " " + pd.Series(last, index=df.index).astype(str)).str.strip()
            out["pitcher_name"] = name.mask(name.eq(""), pd.NA)

    out["source_name"] = source_name
    out["game_date"] = parse_game_date(out["game_date"])

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["pitcher_id"] = out["pitcher_id"].astype("string").str.strip()
    out["pitcher_name"] = out["pitcher_name"].astype("string").str.strip()
    out["team"] = out["team"].map(normalize_team_name).astype("string")
    out["opponent"] = out["opponent"].map(normalize_team_name).astype("string")

    # Retrosheet duplicate stattype rows
    if out["source_stattype"].notna().any():
        stattype_text = out["source_stattype"].astype("string").str.strip().str.lower()
        keep_mask = stattype_text.isna() | stattype_text.eq("value")
        out = out.loc[keep_mask].copy()

    home_away = normalize_home_away(out["home_away"])
    if out["is_home"].notna().any():
        is_home_text = out["is_home"].astype("string").str.strip().str.lower()
        inferred_from_flag = pd.Series(pd.NA, index=out.index, dtype="string")
        inferred_from_flag[is_home_text.isin(["1", "true", "t", "yes", "y", "h", "home"])] = "home"
        inferred_from_flag[is_home_text.isin(["0", "false", "f", "no", "n", "v", "a", "away", "visitor"])] = "away"
        home_away = home_away.fillna(inferred_from_flag)
    out["home_away"] = home_away

    if out["is_starter"].notna().any():
        out["is_starter"] = coerce_is_starter(out["is_starter"])
    else:
        out["is_starter"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="boolean")

    if out["outs_recorded"].isna().all() and out["innings_pitched"].notna().any():
        out["outs_recorded"] = np.floor(out["innings_pitched"] * 3 + 1e-9)

    if out["innings_pitched"].isna().all() and out["outs_recorded"].notna().any():
        out["innings_pitched"] = out["outs_recorded"] / 3.0

    if out["season"].isna().all() and out["game_date"].notna().any():
        out["season"] = out["game_date"].dt.year

    return out, mapping


def build_pitcher_key(df: pd.DataFrame) -> pd.Series:
    pitcher_id = df["pitcher_id"].astype("string").str.strip()
    name_key = (
        df["pitcher_name"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    key = pitcher_id.mask(pitcher_id.isna() | pitcher_id.eq(""), pd.NA)
    return key.fillna("name:" + name_key)


def infer_starter_flag(df: pd.DataFrame) -> pd.Series:
    existing = df["is_starter"].astype("boolean")
    if existing.notna().all():
        return existing

    inferred = existing.copy()
    if "game_id" in df.columns and df["game_id"].notna().any():
        order_key = [c for c in ["game_date", "pitches", "batters_faced", "innings_pitched"] if c in df.columns]
        sort_df = df.copy()
        if "pitches" in order_key:
            ascending = [True if c != "pitches" else False for c in order_key]
            sort_df = sort_df.sort_values(order_key, ascending=ascending, na_position="last")
        else:
            sort_df = sort_df.sort_values(order_key, na_position="last")
        first_idx = sort_df.groupby(["game_id", "team"], dropna=False).head(1).index
        inferred.loc[first_idx] = inferred.loc[first_idx].fillna(True)
        inferred = inferred.fillna(False)
        return inferred.astype(bool)

    if df["innings_pitched"].notna().any():
        inferred = inferred.fillna(df["innings_pitched"] >= 4.0)
    elif df["batters_faced"].notna().any():
        inferred = inferred.fillna(df["batters_faced"] >= 15)
    else:
        inferred = inferred.fillna(True)
    return inferred.astype(bool)


def dedupe_games(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["game_date", "pitches", "batters_faced", "innings_pitched"] if c in df.columns]
    if sort_cols:
        ascending = [True if c == "game_date" else False for c in sort_cols]
        ordered = df.sort_values(sort_cols, ascending=ascending, na_position="last").copy()
    else:
        ordered = df.copy()

    if df["game_id"].notna().any():
        subset = ["pitcher_key", "game_id"]
    else:
        subset = [c for c in ["pitcher_key", "game_date", "team", "opponent"] if c in ordered.columns]
    return ordered.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def join_crosswalk(df: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Optional lightweight join for mlbam/player IDs. Retrosheet IDs will mostly miss."""
    if crosswalk.empty:
        return df
    cw = crosswalk.copy()
    out = df.copy()

    candidate_keys = [c for c in ["mlbam_id", "player_id"] if c in cw.columns]
    if not candidate_keys:
        return out

    for key in candidate_keys:
        temp = cw.copy()
        temp[key] = temp[key].astype("string").str.strip()
        if "player_name" in temp.columns:
            temp = temp.rename(columns={"player_name": "crosswalk_player_name"})
        keep_cols = [c for c in ["player_key", key, "crosswalk_player_name"] if c in temp.columns]
        temp = temp[keep_cols].drop_duplicates(subset=[key])

        left = out["pitcher_id"].astype("string").str.strip()
        merged = out.assign(__join_key=left).merge(
            temp.rename(columns={key: "__join_key"}),
            on="__join_key",
            how="left",
        )
        if "player_key" in merged.columns:
            merged["pitcher_key"] = merged["player_key"].fillna(merged["pitcher_key"])
        if "crosswalk_player_name" in merged.columns:
            merged["pitcher_name"] = merged["pitcher_name"].fillna(merged["crosswalk_player_name"])
        drop_cols = [c for c in ["__join_key", "player_key", "crosswalk_player_name"] if c in merged.columns]
        out = merged.drop(columns=drop_cols)
        # no need to keep attempting if we enriched at least something
        if "pitcher_name" in out.columns and out["pitcher_name"].notna().any():
            break
    return out


def enrich_with_allplayers(df: pd.DataFrame, allplayers: pd.DataFrame) -> pd.DataFrame:
    """Join Retrosheet allplayers.csv to fill pitcher_name and pitcher_hand using retrosheet id + season."""
    if allplayers.empty:
        return df
    ap = allplayers.copy()
    if "id" not in ap.columns:
        return df

    ap["id"] = ap["id"].astype("string").str.strip()
    if "season" in ap.columns:
        ap["season"] = pd.to_numeric(ap["season"], errors="coerce")
    else:
        ap["season"] = pd.NA

    first = ap["first"].fillna("").astype(str).str.strip() if "first" in ap.columns else pd.Series("", index=ap.index)
    last = ap["last"].fillna("").astype(str).str.strip() if "last" in ap.columns else pd.Series("", index=ap.index)
    ap["allplayers_name"] = (first + " " + last).str.strip().replace("", pd.NA)
    if "throw" in ap.columns:
        ap["allplayers_throw"] = ap["throw"].astype("string").str.strip().str.upper()
    keep = [c for c in ["id", "season", "allplayers_name", "allplayers_throw"] if c in ap.columns]
    ap = ap[keep].drop_duplicates(subset=["id", "season"], keep="last")

    out = df.copy()
    out["pitcher_id"] = out["pitcher_id"].astype("string").str.strip()

    # try season-aware join first
    if "season" in out.columns and ap["season"].notna().any():
        merged = out.merge(
            ap,
            how="left",
            left_on=["pitcher_id", "season"],
            right_on=["id", "season"],
        )
    else:
        merged = out.merge(ap.drop(columns=["season"], errors="ignore").drop_duplicates(subset=["id"]), how="left", left_on="pitcher_id", right_on="id")

    merged["pitcher_name"] = merged["pitcher_name"].fillna(merged.get("allplayers_name"))
    if "pitcher_hand" in merged.columns:
        merged["pitcher_hand"] = merged["pitcher_hand"].fillna(merged.get("allplayers_throw"))
    else:
        merged["pitcher_hand"] = merged.get("allplayers_throw")

    drop_cols = [c for c in ["id", "allplayers_name", "allplayers_throw"] if c in merged.columns]
    return merged.drop(columns=drop_cols)


def compute_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["pitcher_key", "game_date"]).copy()
    prior_date = out.groupby("pitcher_key")["game_date"].shift(1)
    days = (out["game_date"] - prior_date).dt.days
    # baseball convention: days rest excludes current day and previous appearance day gap.
    days = days - 1
    days = days.where(days >= 0)
    if "days_rest" in out.columns:
        out["days_rest"] = out["days_rest"].fillna(days)
    else:
        out["days_rest"] = days
    return out


def filter_seasons(df: pd.DataFrame, seasons: Optional[Sequence[int]] = None, min_season: Optional[int] = None) -> pd.DataFrame:
    out = df
    if seasons:
        out = out[out["season"].isin(seasons)].copy()
    if min_season is not None:
        out = out[out["season"] >= min_season].copy()
    return out


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    hand_counts = {}
    if "pitcher_hand" in df.columns:
        hand_counts = df["pitcher_hand"].fillna("NA").astype(str).value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "pitchers": int(df["pitcher_key"].nunique(dropna=True)),
        "date_min": None if df["game_date"].isna().all() else str(df["game_date"].min().date()),
        "date_max": None if df["game_date"].isna().all() else str(df["game_date"].max().date()),
        "starter_rows": int(df["is_starter"].fillna(False).sum()),
        "teams": int(df["team"].dropna().nunique()),
        "named_rows": int(df["pitcher_name"].notna().sum()) if "pitcher_name" in df.columns else 0,
        "handed_rows": int(df["pitcher_hand"].notna().sum()) if "pitcher_hand" in df.columns else 0,
        "days_rest_rows": int(df["days_rest"].notna().sum()) if "days_rest" in df.columns else 0,
        "sources": sorted(df["source_name"].dropna().astype(str).unique().tolist()),
        "hand_counts": hand_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QEPC MLB canonical pitcher game-log spine with allplayers enrichment")
    parser.add_argument("--sources", nargs="+", required=True, help="Raw pitcher log CSV/Parquet files to merge")
    parser.add_argument("--crosswalk", default="cache/mlb/ids/player_crosswalk.parquet", help="Optional crosswalk parquet/csv")
    parser.add_argument("--allplayers", default="cache/mlb/raw/retrosheet/allplayers.csv", help="Optional Retrosheet allplayers.csv for names/hand")
    parser.add_argument("--out", default="cache/mlb/processed/pitcher_game_logs.parquet", help="Canonical output parquet")
    parser.add_argument("--also_csv", action="store_true", help="Also save CSV next to parquet")
    parser.add_argument("--season", nargs="*", type=int, default=None, help="Optional explicit season filter")
    parser.add_argument("--min_season", type=int, default=None, help="Optional minimum season filter")
    parser.add_argument("--only_starters", action="store_true", help="Keep only starting pitcher rows")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    if PATHS is not None:
        PATHS.ensure()

    progress_on = not args.no_progress
    frames: List[pd.DataFrame] = []
    schema_map: Dict[str, Dict[str, str]] = {}

    for src in iter_progress(args.sources, enabled=progress_on, desc="Loading spine sources"):
        path = Path(src)
        raw = load_any(path)
        norm, mapping = normalize_frame(raw, path.name)
        schema_map[path.name] = mapping
        frames.append(norm)

    if not frames:
        raise ValueError("No sources were loaded.")

    print("Combining normalized source tables...")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["game_date"].notna()].copy()
    df = df[df["pitcher_id"].notna() | df["pitcher_name"].notna()].copy()
    df["pitcher_key"] = build_pitcher_key(df)

    crosswalk_path = Path(args.crosswalk)
    if crosswalk_path.exists():
        print(f"Joining crosswalk: {crosswalk_path}")
        crosswalk = load_any(crosswalk_path)
        df = join_crosswalk(df, crosswalk)

    allplayers_path = Path(args.allplayers)
    if allplayers_path.exists():
        print(f"Joining allplayers: {allplayers_path}")
        allplayers = load_any(allplayers_path)
        df = enrich_with_allplayers(df, allplayers)

    print("Inferring starter flags, deduplicating, and computing days rest...")
    df["is_starter"] = infer_starter_flag(df)
    if args.only_starters:
        df = df[df["is_starter"].fillna(False)].copy()

    df = filter_seasons(df, seasons=args.season, min_season=args.min_season)
    df = dedupe_games(df)
    df = df.sort_values(["game_date", "pitcher_key"]).reset_index(drop=True)
    df = compute_days_rest(df)

    final_columns = [
        "game_date", "season", "game_id", "pitcher_key", "pitcher_id", "pitcher_name",
        "team", "opponent", "home_away", "is_starter", "pitcher_hand",
        "innings_pitched", "outs_recorded", "strikeouts", "walks", "hits_allowed",
        "earned_runs", "runs_allowed", "batters_faced", "pitches", "days_rest",
        "decision", "source_name",
    ]
    final_columns = [c for c in final_columns if c in df.columns]
    out_df = df[final_columns].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary = summarize(out_df)
    report_path = out_path.with_suffix(".summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "source_column_mapping": schema_map}, f, indent=2)

    print(f"saved parquet: {out_path}")
    print(f"saved summary: {report_path}")
    print(json.dumps(summary, indent=2))

    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        out_df.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")


if __name__ == "__main__":
    main()
