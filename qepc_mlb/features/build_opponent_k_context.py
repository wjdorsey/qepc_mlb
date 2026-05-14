
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SAFE_OPP_PREFIXES = (
    "opp_team_games_prior",
    "opp_team_days_since_game",
    "opp_team_ks_lag1",
    "opp_team_ks_roll",
    "opp_team_ks_std",
    "opp_team_ks_expanding_mean",
    "opp_team_starter_ks_lag1",
    "opp_team_starter_ks_roll",
    "opp_team_k_per_bf_lag1",
    "opp_team_k_per_bf_roll",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-safe opponent strikeout context features from pitcher game logs.")
    p.add_argument("--input", required=True, help="Path to pitcher game logs parquet/csv (ideally enriched logs).")
    p.add_argument("--out", default="cache/mlb/processed/pitcher_game_logs_enriched_opp.parquet", help="Output parquet/csv path.")
    p.add_argument("--min_season", type=int, default=None, help="Optional minimum season filter before building context.")
    p.add_argument("--also_csv", action="store_true", help="Also write CSV alongside parquet.")
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


def maybe_tqdm(it: Iterable, *, total: int | None, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(it, total=total, desc=desc)
    return it


def coerce_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False
    txt = series.astype("string").str.strip().str.lower()
    out.loc[out.isna() & txt.isin(["true", "t", "yes", "y", "starter", "sp"])] = True
    out.loc[out.isna() & txt.isin(["false", "f", "no", "n", "reliever", "rp"])] = False
    return out


def normalize_team(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    aliases = {
        "KCA": "KC",
        "KAN": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "TBD": "TB",
        "WSN": "WSH",
        "WAS": "WSH",
        "AZ": "ARI",
        "CHW": "CWS",
        "NYN": "NYM",
        "NYA": "NYY",
        "LAN": "LAD",
        "ANA": "LAA",
        "OAK": "ATH",
        "SAC": "ATH",
    }
    return s.map(lambda x: aliases.get(x, x) if x is not pd.NA else x)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "is_starter" in out.columns:
        out["is_starter"] = coerce_bool(out["is_starter"]).fillna(False).astype(bool)
    else:
        out["is_starter"] = False
    for c in ["strikeouts", "batters_faced", "outs_recorded", "innings_pitched"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "opponent" in out.columns:
        out["opponent"] = normalize_team(out["opponent"])
    if "team" in out.columns:
        out["team"] = normalize_team(out["team"])
    if "game_id" not in out.columns:
        out["game_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    else:
        out["game_id"] = out["game_id"].astype("string")
    return out


def add_team_game_context(df: pd.DataFrame, show_progress: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if "opponent" not in work.columns:
        raise SystemExit("Expected 'opponent' column in game logs.")
    work = work[work["game_date"].notna() & work["opponent"].notna()].copy()

    grouping_keys = ["season", "game_date", "opponent"]
    if work["game_id"].notna().any():
        grouping_keys.append("game_id")

    agg_dict = {"strikeouts": "sum"}
    if "batters_faced" in work.columns:
        agg_dict["batters_faced"] = "sum"

    team_games = (
        work.groupby(grouping_keys, dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={
            "opponent": "opp_team",
            "strikeouts": "opp_team_ks_game",
            "batters_faced": "opp_team_bf_game",
        })
    )

    starter_mask = work["is_starter"].fillna(False).astype(bool)
    if starter_mask.any():
        starter_games = (
            work.loc[starter_mask]
            .groupby(grouping_keys, dropna=False)["strikeouts"]
            .sum()
            .reset_index()
            .rename(columns={"opponent": "opp_team", "strikeouts": "opp_team_starter_ks_game"})
        )
        team_games = team_games.merge(
            starter_games,
            how="left",
            on=[k if k != "opponent" else "opp_team" for k in grouping_keys],
        )
    else:
        team_games["opp_team_starter_ks_game"] = np.nan

    sort_cols = ["game_date"] + (["game_id"] if "game_id" in team_games.columns else [])
    team_games = team_games.sort_values(["opp_team"] + sort_cols).reset_index(drop=True)

    pieces = []
    iterator = maybe_tqdm(
        team_games.groupby("opp_team", sort=False),
        total=int(team_games["opp_team"].nunique()),
        desc="Opponent context rolls",
        enabled=show_progress,
    )
    for _, g in iterator:
        g = g.sort_values(sort_cols).copy()
        g["opp_team_games_prior"] = np.arange(len(g))
        g["opp_team_days_since_game"] = g["game_date"].diff().dt.days

        g["opp_team_ks_lag1"] = g["opp_team_ks_game"].shift(1)
        for w in [3, 5, 10]:
            g[f"opp_team_ks_roll{w}"] = g["opp_team_ks_game"].shift(1).rolling(w, min_periods=1).mean()
            g[f"opp_team_ks_std{w}"] = g["opp_team_ks_game"].shift(1).rolling(w, min_periods=2).std()

        g["opp_team_ks_expanding_mean"] = g["opp_team_ks_game"].shift(1).expanding(min_periods=1).mean()
        g["opp_team_starter_ks_lag1"] = g["opp_team_starter_ks_game"].shift(1)
        for w in [3, 5, 10]:
            g[f"opp_team_starter_ks_roll{w}"] = g["opp_team_starter_ks_game"].shift(1).rolling(w, min_periods=1).mean()

        if "opp_team_bf_game" in g.columns:
            g["opp_team_k_per_bf_game"] = np.where(g["opp_team_bf_game"] > 0, g["opp_team_ks_game"] / g["opp_team_bf_game"], np.nan)
            g["opp_team_k_per_bf_lag1"] = g["opp_team_k_per_bf_game"].shift(1)
            for w in [3, 5, 10]:
                g[f"opp_team_k_per_bf_roll{w}"] = g["opp_team_k_per_bf_game"].shift(1).rolling(w, min_periods=1).mean()

        pieces.append(g)

    ctx = pd.concat(pieces, ignore_index=True)
    return work, ctx


def safe_opp_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if any(c.startswith(prefix) for prefix in SAFE_OPP_PREFIXES):
            out.append(c)
    return out


def join_context(game_logs: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    left = game_logs.copy()
    merge_keys_left = ["season", "game_date", "opponent"]
    merge_keys_right = ["season", "game_date", "opp_team"]
    if "game_id" in ctx.columns and left["game_id"].notna().any():
        merge_keys_left.append("game_id")
        merge_keys_right.append("game_id")

    safe_cols = safe_opp_columns(ctx)
    ctx_small = ctx[merge_keys_right + safe_cols].drop_duplicates()
    merged = left.merge(ctx_small, how="left", left_on=merge_keys_left, right_on=merge_keys_right)
    if "opp_team" in merged.columns:
        merged = merged.drop(columns=["opp_team"])
    return merged


def build_summary(df: pd.DataFrame) -> dict:
    opp_cols = [c for c in df.columns if c.startswith("opp_team_")]
    nonnull = {c: int(df[c].notna().sum()) for c in opp_cols}
    return {
        "rows": int(len(df)),
        "pitchers": int(df["pitcher_id"].nunique()) if "pitcher_id" in df.columns else None,
        "date_min": None if "game_date" not in df.columns or df["game_date"].dropna().empty else str(df["game_date"].min().date()),
        "date_max": None if "game_date" not in df.columns or df["game_date"].dropna().empty else str(df["game_date"].max().date()),
        "opponent_context_columns": opp_cols,
        "opponent_context_nonnull": nonnull,
        "rows_with_any_opp_context": int(df[opp_cols].notna().any(axis=1).sum()) if opp_cols else 0,
        "rows_with_roll5": int(df["opp_team_ks_roll5"].notna().sum()) if "opp_team_ks_roll5" in df.columns else 0,
    }


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.out)

    print("Loading pitcher game logs...")
    df = load_any(inp)
    print("Normalizing schema...")
    df = normalize(df)

    if args.min_season is not None and "season" in df.columns:
        df = df[df["season"].fillna(-1).astype(int) >= args.min_season].copy()

    print("Building leakage-safe opponent strikeout context from prior team games...")
    base_df, ctx = add_team_game_context(df, show_progress=not args.no_progress)
    print("Joining opponent context back to pitcher game logs...")
    merged = join_context(base_df, ctx)

    save_any(merged, out, also_csv=args.also_csv)
    summary = build_summary(merged)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
