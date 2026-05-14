from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ID_ALIASES = {
    "mlbam_id": ["mlbam_id", "mlb_id", "mlbid", "player_mlbam_id", "batter", "pitcher"],
    "fangraphs_id": ["fangraphs_id", "fg_id", "fgid", "playerid"],
    "retrosheet_id": ["retrosheet_id", "retro_id", "retroid"],
    "bbref_id": ["bbref_id", "baseball_reference_id", "bref_id"],
    "lahman_id": ["lahman_id", "playerid_lahman"],
    "player_id": ["player_id", "id"],
    "player_name": ["player_name", "name", "full_name", "player", "Name"],
    "first_name": ["first_name", "firstname", "name_first"],
    "last_name": ["last_name", "lastname", "name_last"],
    "team": ["team", "team_name", "team_abbr", "club"],
}


def load_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def first_existing(cols: Iterable[str], aliases: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def normalize_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = pd.DataFrame()
    for canon, aliases in ID_ALIASES.items():
        col = first_existing(df.columns, aliases)
        out[canon] = df[col] if col else pd.Series([pd.NA] * len(df))

    if out["player_name"].isna().all():
        first = out["first_name"].fillna("").astype(str).str.strip()
        last = out["last_name"].fillna("").astype(str).str.strip()
        name = (first + " " + last).str.strip()
        out["player_name"] = name.mask(name.eq(""), pd.NA)

    out["source_name"] = source_name
    return out


def build_player_key(df: pd.DataFrame) -> pd.Series:
    preferred = ["mlbam_id", "fangraphs_id", "retrosheet_id", "bbref_id", "lahman_id", "player_id"]
    key = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in preferred:
        vals = df[col].astype("string").str.strip()
        mask = key.isna() & vals.notna() & vals.ne("") & vals.ne("<NA>")
        key.loc[mask] = col + ":" + vals.loc[mask]
    fallback_name = (
        df["player_name"].fillna("unknown").astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    key = key.fillna("name:" + fallback_name)
    return key


def iter_progress(items, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical MLB player ID crosswalk")
    parser.add_argument("--sources", nargs="+", required=True, help="CSV/Parquet sources to merge")
    parser.add_argument("--out", default="cache/mlb/ids/player_crosswalk.parquet")
    parser.add_argument("--also_csv", action="store_true")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    show_progress = not args.no_progress

    frames = []
    for src in iter_progress(args.sources, enabled=show_progress, desc="Loading ID sources"):
        path = Path(src)
        df = load_any(path)
        frames.append(normalize_frame(df, path.name))

    if show_progress:
        print("Normalizing and combining source tables...")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates()
    merged["player_key"] = build_player_key(merged)

    agg_cols = [c for c in merged.columns if c != "source_name"]
    if show_progress:
        print("Building canonical player keys and crosswalk...")
    crosswalk = (
        merged.sort_values(["player_key", "source_name"])
        .groupby("player_key", as_index=False)[agg_cols]
        .first()
    )
    sources = merged.groupby("player_key")["source_name"].agg(lambda s: sorted(set(s)))
    crosswalk = crosswalk.merge(sources.rename("sources"), on="player_key", how="left")
    crosswalk = crosswalk.sort_values("player_name", na_position="last").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(out_path, index=False)
    print(f"saved parquet: {out_path} ({len(crosswalk):,} players)")

    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        crosswalk.to_csv(csv_path, index=False)
        print(f"saved csv: {csv_path}")


if __name__ == "__main__":
    main()
