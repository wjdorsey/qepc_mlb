from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATE_CANDIDATES = ["game_date", "date", "gamedate"]
GAME_ID_CANDIDATES = ["game_id", "gid", "gameid"]
TEAM_CANDIDATES = ["team", "bat_team", "batting_team", "club", "tm"]
OPP_CANDIDATES = ["opponent", "opp", "vs", "opp_team"]
STATTYPE_CANDIDATES = ["stattype", "stat_type"]
BATTING_K_CANDIDATES = ["b_k", "batting_k", "bat_k", "strikeouts", "so"]
BATTING_PA_CANDIDATES = ["b_pa", "pa", "batting_pa"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lineup-level strikeout context from Retrosheet teamstats and attach it to pitcher game logs.")
    p.add_argument("--input", required=True, help="Pitcher game logs parquet/csv, ideally the current enriched_opp_env table.")
    p.add_argument("--teamstats", required=True, help="Retrosheet teamstats.csv path.")
    p.add_argument("--out", required=True, help="Output parquet path.")
    p.add_argument("--also_csv", action="store_true")
    p.add_argument("--min_season", type=int, default=None)
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()



def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")



def first_present(cols: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None



def parse_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, errors="coerce")
    mask = dt.isna() & s.str.fullmatch(r"\d{8}", na=False)
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s.loc[mask], format="%Y%m%d", errors="coerce")
    return dt



def normalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)
    game_id_col = first_present(cols, GAME_ID_CANDIDATES)
    if game_id_col and game_id_col != "game_id":
        out["game_id"] = out[game_id_col]
    date_col = first_present(cols, DATE_CANDIDATES)
    if date_col and date_col != "game_date":
        out["game_date"] = out[date_col]
    out["game_date"] = parse_date(out["game_date"])
    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    for col in ["game_id", "pitcher_id", "team", "opponent", "pitcher_name", "pitcher_hand"]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out



def find_lineup_cols(cols: list[str]) -> list[str]:
    found = []
    for c in cols:
        cl = c.lower()
        if re.fullmatch(r"start_l\d+", cl) or re.fullmatch(r"start\d+", cl) or re.fullmatch(r"lineup_\d+", cl):
            found.append(c)
    return sorted(found, key=lambda x: int(re.findall(r"(\d+)", x)[-1]))



def normalize_teamstats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)

    stattype_col = first_present(cols, STATTYPE_CANDIDATES)
    if stattype_col:
        keep = out[stattype_col].astype("string").str.lower().eq("value")
        if keep.any():
            out = out.loc[keep].copy()

    game_id_col = first_present(list(out.columns), GAME_ID_CANDIDATES)
    if game_id_col and game_id_col != "game_id":
        out["game_id"] = out[game_id_col]

    date_col = first_present(list(out.columns), DATE_CANDIDATES)
    if date_col and date_col != "game_date":
        out["game_date"] = out[date_col]
    out["game_date"] = parse_date(out["game_date"])

    team_col = first_present(list(out.columns), TEAM_CANDIDATES)
    opp_col = first_present(list(out.columns), OPP_CANDIDATES)
    k_col = first_present(list(out.columns), BATTING_K_CANDIDATES)
    pa_col = first_present(list(out.columns), BATTING_PA_CANDIDATES)
    line_cols = find_lineup_cols(list(out.columns))

    if team_col is None or k_col is None:
        raise SystemExit(
            f"Could not find required teamstats columns. team_col={team_col}, batting_k_col={k_col}. Available columns include: {list(out.columns)[:40]}"
        )

    out["batting_team"] = out[team_col].astype("string").str.strip()
    out["opp_team"] = out[opp_col].astype("string").str.strip() if opp_col else pd.NA
    out["season"] = pd.to_numeric(out.get("season", out["game_date"].dt.year), errors="coerce").astype("Int64")
    out["lineup_batting_ks"] = pd.to_numeric(out[k_col], errors="coerce")
    if pa_col and pa_col in out.columns:
        out["lineup_batting_pa"] = pd.to_numeric(out[pa_col], errors="coerce")
    else:
        out["lineup_batting_pa"] = np.nan

    for lc in line_cols:
        out[lc] = out[lc].astype("string").str.strip()
    out["lineup_known_count"] = out[line_cols].notna().sum(axis=1).astype(float) if line_cols else np.nan

    # Dedupe any repeated team-game rows.
    keep_cols = ["game_id", "game_date", "season", "batting_team", "opp_team", "lineup_batting_ks", "lineup_batting_pa", "lineup_known_count"] + line_cols
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].drop_duplicates(subset=[c for c in ["game_id", "batting_team"] if c in keep_cols]).copy()
    return out



def add_lineup_rolls(teamstats: pd.DataFrame, show_progress: bool) -> pd.DataFrame:
    work = teamstats.sort_values(["batting_team", "game_date", "game_id"]).copy()
    grouped = work.groupby("batting_team", sort=False)
    iterator = grouped if not show_progress else tqdm(grouped, total=work["batting_team"].nunique(), desc="Rolling lineup K context")
    pieces: list[pd.DataFrame] = []
    line_cols = find_lineup_cols(list(work.columns))

    for _, g in iterator:
        g = g.sort_values(["game_date", "game_id"]).copy()
        g["lineup_team_ks_lag1"] = g["lineup_batting_ks"].shift(1)
        for w in [3, 5, 10]:
            g[f"lineup_team_ks_roll{w}"] = g["lineup_batting_ks"].shift(1).rolling(w, min_periods=1).mean()
        g["lineup_team_ks_expanding_mean"] = g["lineup_batting_ks"].shift(1).expanding(min_periods=1).mean()
        g["lineup_team_ks_std5"] = g["lineup_batting_ks"].shift(1).rolling(5, min_periods=2).std()

        if "lineup_batting_pa" in g.columns:
            lag_pa = g["lineup_batting_pa"].shift(1)
            g["lineup_team_k_per_pa_lag1"] = np.where(lag_pa > 0, g["lineup_batting_ks"].shift(1) / lag_pa, np.nan)
            g["lineup_team_k_per_pa_roll5"] = g["lineup_team_k_per_pa_lag1"].rolling(5, min_periods=1).mean()
        else:
            g["lineup_team_k_per_pa_lag1"] = np.nan
            g["lineup_team_k_per_pa_roll5"] = np.nan

        if "lineup_known_count" in g.columns:
            g["lineup_known_count_lag1"] = g["lineup_known_count"].shift(1)
            g["lineup_known_count_roll5"] = g["lineup_known_count"].shift(1).rolling(5, min_periods=1).mean()
        else:
            g["lineup_known_count_lag1"] = np.nan
            g["lineup_known_count_roll5"] = np.nan

        if line_cols:
            prev_sets: list[set[str]] = []
            overlaps: list[float] = []
            for _, row in g.iterrows():
                cur = {str(row[c]) for c in line_cols if pd.notna(row[c]) and str(row[c]).strip() != ""}
                if prev_sets:
                    overlaps.append(float(len(cur & prev_sets[-1])))
                else:
                    overlaps.append(np.nan)
                prev_sets.append(cur)
            g["lineup_returning_starters_prev"] = overlaps
            g["lineup_returning_starters_roll5"] = pd.Series(overlaps, index=g.index).shift(0).rolling(5, min_periods=1).mean()
        else:
            g["lineup_returning_starters_prev"] = np.nan
            g["lineup_returning_starters_roll5"] = np.nan

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)



def attach_to_logs(logs: pd.DataFrame, lineup: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    right_cols = [
        "game_id", "game_date", "batting_team",
    ] + [c for c in lineup.columns if c.startswith("lineup_")]
    right = lineup[right_cols].copy()

    merged = logs.merge(
        right,
        how="left",
        left_on=["game_id", "opponent"],
        right_on=["game_id", "batting_team"],
        suffixes=("", "_lineup"),
    )
    matched = merged[[c for c in merged.columns if c.startswith("lineup_")]].notna().any(axis=1)
    if matched.mean() >= 0.95:
        merged["has_lineup_context"] = matched
        return merged.drop(columns=[c for c in ["batting_team"] if c in merged.columns]), "game_id+opponent"

    # Fallback for unmatched rows: date + opponent team.
    fallback_right = lineup.sort_values(["game_date", "batting_team"]).drop_duplicates(subset=["game_date", "batting_team"]).copy()
    fb_cols = ["game_date", "batting_team"] + [c for c in fallback_right.columns if c.startswith("lineup_")]
    fallback_right = fallback_right[fb_cols]

    unmatched = merged.loc[~matched, logs.columns].copy()
    fallback = unmatched.merge(
        fallback_right,
        how="left",
        left_on=["game_date", "opponent"],
        right_on=["game_date", "batting_team"],
        suffixes=("", "_lineup"),
    ).drop(columns=[c for c in ["batting_team"] if c in fallback_right.columns], errors="ignore")

    kept = merged.loc[matched].copy()
    combined = pd.concat([kept, fallback], ignore_index=True, sort=False)
    combined = combined.sort_values(["game_date", "pitcher_id"]).reset_index(drop=True)
    combined["has_lineup_context"] = combined[[c for c in combined.columns if c.startswith("lineup_")]].notna().any(axis=1)
    return combined, "game_id+opponent_then_date+opponent"



def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    teamstats_path = Path(args.teamstats)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pitcher game logs...")
    logs = normalize_logs(load_any(inp))
    print("Loading teamstats...")
    teamstats = normalize_teamstats(load_any(teamstats_path))
    if args.min_season is not None:
        logs = logs[logs["season"].fillna(-1).astype(int) >= args.min_season].copy()
        teamstats = teamstats[teamstats["season"].fillna(-1).astype(int) >= args.min_season].copy()

    print("Building leakage-safe lineup strikeout context...")
    lineup_ctx = add_lineup_rolls(teamstats, show_progress=not args.no_progress)

    print("Joining lineup context to pitcher game logs...")
    combined, matched_by = attach_to_logs(logs, lineup_ctx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"saved parquet: {out_path}")
    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        combined.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")

    summary = {
        "rows": int(len(combined)),
        "seasons": sorted([int(x) for x in combined["season"].dropna().astype(int).unique().tolist()]),
        "matched_by": matched_by,
        "lineup_match_rows": int(combined["has_lineup_context"].fillna(False).sum()),
        "lineup_match_rate": float(combined["has_lineup_context"].fillna(False).mean()),
        "lineup_columns": [c for c in combined.columns if c.startswith("lineup_")],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
