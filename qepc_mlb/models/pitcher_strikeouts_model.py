from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

TARGET_CANDIDATES = ["strikeouts", "pitcher_strikeouts", "so", "k", "ks"]
PITCHER_ID_CANDIDATES = ["pitcher_id", "player_id", "mlbam_id", "pitcher_mlbam_id"]
DATE_CANDIDATES = ["game_date", "date", "game_dt"]
SEASON_CANDIDATES = ["season", "year"]
TEAM_CANDIDATES = ["team", "pitching_team", "player_team"]
OPP_CANDIDATES = ["opponent", "opp", "batting_team", "opponent_team"]
HAND_CANDIDATES = ["pitcher_hand", "throws", "p_throws"]
STARTER_FLAG_CANDIDATES = ["is_starter", "starter", "started", "p_gs"]
BATTERS_FACED_CANDIDATES = ["batters_faced", "bf", "p_bfp"]
INNINGS_PITCHED_CANDIDATES = ["innings_pitched", "ip"]
OUTS_RECORDED_CANDIDATES = ["outs_recorded", "ipouts", "p_ipouts"]
PITCHES_CANDIDATES = ["pitches", "pitch_count"]
DAYS_REST_CANDIDATES = ["days_rest", "rest_days"]
WALKS_CANDIDATES = ["walks", "bb", "p_w"]
HITS_ALLOWED_CANDIDATES = ["hits_allowed", "hits", "h", "p_h"]
ER_CANDIDATES = ["earned_runs", "er", "p_er"]
RUNS_ALLOWED_CANDIDATES = ["runs_allowed", "runs", "r", "p_r"]
SWSTR_CANDIDATES = ["swstr_rate", "swinging_strike_rate", "swstr_pct"]
CSW_CANDIDATES = ["csw_rate", "csw_pct"]
OPP_K_CANDIDATES = ["opp_k_rate", "opponent_k_rate", "lineup_k_rate"]
HOME_AWAY_CANDIDATES = ["home_away", "vishome"]


NUMERIC_BASE_FEATURES = [
    "batters_faced",
    "innings_pitched",
    "outs_recorded",
    "pitches",
    "days_rest",
    "walks",
    "hits_allowed",
    "earned_runs",
    "runs_allowed",
    "swstr_rate",
    "csw_rate",
    "opp_k_rate",
]


def first_existing(columns: Iterable[str], candidates: List[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


# noqa: C901 - explicit mapping is easier to maintain here

def normalize_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    cols = list(df.columns)

    def grab(name: str, candidates: List[str], required: bool = False) -> None:
        col = first_existing(cols, candidates)
        if required and col is None:
            raise ValueError(f"Could not find required column for {name}. Candidates: {candidates}")
        if col is not None:
            mapping[name] = str(col)

    grab("target", TARGET_CANDIDATES, required=True)
    grab("pitcher_id", PITCHER_ID_CANDIDATES, required=True)
    grab("game_date", DATE_CANDIDATES, required=True)
    grab("season", SEASON_CANDIDATES)
    grab("team", TEAM_CANDIDATES)
    grab("opponent", OPP_CANDIDATES)
    grab("pitcher_hand", HAND_CANDIDATES)
    grab("is_starter", STARTER_FLAG_CANDIDATES)
    grab("batters_faced", BATTERS_FACED_CANDIDATES)
    grab("innings_pitched", INNINGS_PITCHED_CANDIDATES)
    grab("outs_recorded", OUTS_RECORDED_CANDIDATES)
    grab("pitches", PITCHES_CANDIDATES)
    grab("days_rest", DAYS_REST_CANDIDATES)
    grab("walks", WALKS_CANDIDATES)
    grab("hits_allowed", HITS_ALLOWED_CANDIDATES)
    grab("earned_runs", ER_CANDIDATES)
    grab("runs_allowed", RUNS_ALLOWED_CANDIDATES)
    grab("swstr_rate", SWSTR_CANDIDATES)
    grab("csw_rate", CSW_CANDIDATES)
    grab("opp_k_rate", OPP_K_CANDIDATES)
    grab("home_away", HOME_AWAY_CANDIDATES)

    out = pd.DataFrame(index=df.index)
    for std_name, src_col in mapping.items():
        out[std_name] = df[src_col]

    out["pitcher_id"] = out["pitcher_id"].astype("string").str.strip()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")

    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
    else:
        out["season"] = out["game_date"].dt.year

    for col in NUMERIC_BASE_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "outs_recorded" not in out.columns and "innings_pitched" in out.columns:
        out["outs_recorded"] = pd.to_numeric(out["innings_pitched"], errors="coerce") * 3.0
    if "innings_pitched" not in out.columns and "outs_recorded" in out.columns:
        out["innings_pitched"] = pd.to_numeric(out["outs_recorded"], errors="coerce") / 3.0

    if "is_starter" in out.columns:
        out["is_starter"] = coerce_is_starter(out["is_starter"])
    else:
        out["is_starter"] = pd.Series(True, index=out.index, dtype="boolean")

    if "home_away" in out.columns:
        home_map = {"h": "home", "home": "home", "v": "away", "a": "away", "away": "away"}
        out["home_away"] = out["home_away"].astype("string").str.strip().str.lower().map(home_map).fillna(out["home_away"].astype("string"))

    for col in ["team", "opponent", "pitcher_hand"]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out.loc[out[col].isin(["", "<NA>", "nan", "None"]), col] = pd.NA

    out = out[out["pitcher_id"].notna() & out["game_date"].notna() & out["target"].notna()].copy()
    return out, mapping


def coerce_is_starter(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[numeric == 1] = True
    out.loc[numeric == 0] = False

    text = series.astype("string").str.strip().str.lower()
    out.loc[out.isna() & text.isin(["sp", "starter", "start", "true", "t", "yes", "y"])] = True
    out.loc[out.isna() & text.isin(["rp", "reliever", "false", "f", "no", "n"])] = False
    return out.fillna(False).astype("boolean")


def compute_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    if "days_rest" not in df.columns:
        df["days_rest"] = np.nan
    if df["days_rest"].notna().any():
        return df

    df = df.sort_values(["pitcher_id", "game_date"]).copy()
    prev_dates = df.groupby("pitcher_id", group_keys=False)["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - prev_dates).dt.days
    return df


def add_rolling_features(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    df = df.sort_values(["pitcher_id", "game_date", "team", "opponent"], na_position="last").copy()
    parts: List[pd.DataFrame] = []
    grouped = list(df.groupby("pitcher_id", sort=False))
    iterator = grouped
    if show_progress and tqdm is not None:
        iterator = tqdm(grouped, desc="Rolling pitcher features", total=len(grouped))

    for _, g in iterator:
        g = g.sort_values("game_date").copy()
        g["games_prior"] = np.arange(len(g))
        g["target_lag1"] = g["target"].shift(1)

        for window in [3, 5, 10]:
            shifted_target = g["target"].shift(1)
            g[f"k_roll_mean_{window}"] = shifted_target.rolling(window, min_periods=1).mean()
            g[f"k_roll_std_{window}"] = shifted_target.rolling(window, min_periods=2).std()

        for base_col in NUMERIC_BASE_FEATURES:
            if base_col in g.columns:
                shifted = g[base_col].shift(1)
                g[f"{base_col}_lag1"] = shifted
                g[f"{base_col}_roll5"] = shifted.rolling(5, min_periods=1).mean()

        if "batters_faced" in g.columns:
            bf_shift = g["batters_faced"].shift(1).replace(0, np.nan)
            g["k_per_bf_lag1"] = g["target"].shift(1) / bf_shift
            numer = g["target"].shift(1).rolling(5, min_periods=1).sum()
            denom = g["batters_faced"].shift(1).rolling(5, min_periods=1).sum().replace(0, np.nan)
            g["k_per_bf_roll5"] = numer / denom

        parts.append(g)

    return pd.concat(parts, ignore_index=True) if parts else df.copy()


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("game_date")
    cut = max(1, int(len(ordered) * (1.0 - test_size)))
    train_df = ordered.iloc[:cut].copy()
    test_df = ordered.iloc[cut:].copy()
    if test_df.empty:
        raise ValueError("Test split is empty. Increase data size or reduce filters.")
    return train_df, test_df


def prune_feature_lists(train_df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> Tuple[List[str], List[str], List[str]]:
    dropped: List[str] = []

    keep_numeric: List[str] = []
    for col in numeric_features:
        if train_df[col].notna().any():
            keep_numeric.append(col)
        else:
            dropped.append(col)

    keep_categorical: List[str] = []
    for col in categorical_features:
        vals = train_df[col].dropna().astype("string").str.strip()
        if not vals.empty and (vals != "").any():
            keep_categorical.append(col)
        else:
            dropped.append(col)

    return keep_numeric, keep_categorical, dropped


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ))
    if not transformers:
        raise ValueError("No usable features remain after pruning empty columns.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "bias": bias}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QEPC MLB pitcher strikeouts model")
    parser.add_argument("--input", required=True, help="Pitcher game logs csv/parquet")
    parser.add_argument("--out_dir", default="artifacts/mlb/models/pitcher_strikeouts")
    parser.add_argument("--predictions_out", default="artifacts/mlb/predictions/pitcher_strikeouts_predictions.parquet")
    parser.add_argument("--min_games", type=int, default=5)
    parser.add_argument("--min_season", type=int, default=None)
    parser.add_argument("--max_season", type=int, default=None)
    parser.add_argument("--starters_only", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    show_progress = not args.no_progress

    if show_progress:
        print("Loading pitcher game logs...")
    raw = load_frame(Path(args.input))

    if show_progress:
        print("Normalizing schema...")
    df, mapping = normalize_schema(raw)
    df = compute_days_rest(df)

    if args.min_season is not None:
        df = df[df["season"] >= args.min_season].copy()
    if args.max_season is not None:
        df = df[df["season"] <= args.max_season].copy()
    if args.starters_only:
        df = df[df["is_starter"].fillna(False)].copy()

    if df.empty:
        raise ValueError("No rows remain after season/starter filters.")

    if show_progress:
        summary = {
            "rows": int(len(df)),
            "pitchers": int(df["pitcher_id"].nunique()),
            "date_min": str(df["game_date"].min().date()) if df["game_date"].notna().any() else None,
            "date_max": str(df["game_date"].max().date()) if df["game_date"].notna().any() else None,
            "starter_rows": int(df["is_starter"].fillna(False).sum()),
        }
        print("Input summary:")
        print(json.dumps(summary, indent=2))
        print("Building rolling features...")

    df = add_rolling_features(df, show_progress=show_progress)
    df = df[df["games_prior"] >= args.min_games].copy()
    df = df[df["target"].notna()].copy()

    if df.empty:
        raise ValueError("No rows remain after rolling-feature and min_games filters.")

    excluded = {"target", "pitcher_id", "game_date", "source_name", "game_id", "pitcher_key", "pitcher_name", "season"}
    categorical_features = [c for c in ["team", "opponent", "pitcher_hand", "home_away", "is_starter"] if c in df.columns]
    numeric_features = [
        c for c in df.columns
        if c not in excluded
        and c not in categorical_features
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]

    train_df, test_df = train_test_split_time(df, test_size=args.test_size)
    numeric_features, categorical_features, dropped_features = prune_feature_lists(train_df, numeric_features, categorical_features)

    feature_cols = list(dict.fromkeys(numeric_features + categorical_features))
    if not feature_cols:
        raise ValueError("No usable feature columns remain after pruning empty columns.")

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    pipe = build_pipeline(numeric_features, categorical_features)
    if show_progress:
        print("Training pitcher strikeout model...")
    pipe.fit(X_train, y_train)

    if show_progress:
        print("Generating predictions and saving artifacts...")
    preds = np.clip(pipe.predict(X_test), 0, None)

    metrics = evaluate(y_test, preds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"
    feat_path = out_dir / "feature_columns.json"

    joblib.dump(pipe, model_path)
    metadata = {
        "metrics": metrics,
        "column_mapping": mapping,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "min_games": args.min_games,
        "min_season": args.min_season,
        "max_season": args.max_season,
        "starters_only": bool(args.starters_only),
        "dropped_empty_features": dropped_features,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump({"numeric_features": numeric_features, "categorical_features": categorical_features}, f, indent=2)

    pred_df = test_df[["pitcher_id", "game_date"]].copy()
    for col in ["team", "opponent", "pitcher_name", "season"]:
        if col in test_df.columns:
            pred_df[col] = test_df[col]
    pred_df["actual_strikeouts"] = y_test.values
    pred_df["pred_strikeouts"] = preds
    pred_df.to_parquet(args.predictions_out, index=False)

    print("saved model:", model_path)
    print("saved metadata:", meta_path)
    print("saved predictions:", args.predictions_out)
    if dropped_features:
        print("dropped all-empty features:", ", ".join(dropped_features))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
