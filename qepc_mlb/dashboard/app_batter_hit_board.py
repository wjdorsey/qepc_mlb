#!/usr/bin/env python3
"""
QEPC-MLB Daily Batter 1+ Hit Board Dashboard.

Run with:

  streamlit run qepc_mlb/dashboard/app_batter_hit_board.py

This dashboard reads outputs from:

  qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py

Production board:
  QEPC-MLB Daily Batter 1+ Hit Board v4.1
  base 80% v3 + 20% Savant-light
  v3 top-5 protection gate: 90% v3 + 10% Savant-light
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


APP_TITLE = "QEPC-MLB Daily Batter 1+ Hit Board"
PRED_DIR = Path("artifacts/mlb/predictions/batter_1plus_hit_blend")


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "qepc_mlb").exists():
            return p
    return Path.cwd()


PROJECT_ROOT = find_project_root()


def date_tag_from_iso(iso_date: str) -> str:
    return iso_date.replace("-", "")


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return None

    return None


def latest_file(pattern: str) -> Optional[Path]:
    files = sorted((PROJECT_ROOT / PRED_DIR).glob(pattern))
    return files[-1] if files else None


def run_daily_pipeline(score_date: str, top_n: int, skip_fetch: bool) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        "qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py",
        "--date",
        score_date,
        "--top_n",
        str(top_n),
    ]

    if skip_fetch:
        cmd.append("--skip_fetch")

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    return proc.returncode, proc.stdout, proc.stderr


def probability_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "pred_prob_pct",
        "blend_prob",
        "blend_prob_base_80_20",
        "v3_prob",
        "savant_prob",
    ]
    return [c for c in candidates if c in df.columns]


def format_board(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["v3_prob", "savant_prob", "blend_prob", "blend_prob_base_80_20"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "pred_prob_pct" not in out.columns and "blend_prob" in out.columns:
        out["pred_prob_pct"] = (out["blend_prob"] * 100).round(1)

    # Friendly display order.
    cols = [
        "pred_rank_daily",
        "batter_name",
        "team",
        "opponent",
        "lineup_slot",
        "pred_prob_pct",
        "v3_prob",
        "savant_prob",
        "blend_prob_base_80_20",
        "v3_rank_daily",
        "blend_prob",
    ]

    keep = [c for c in cols if c in out.columns]
    rest = [c for c in out.columns if c not in keep]

    return out[keep + rest]


def make_probability_chart(df: pd.DataFrame) -> None:
    needed = {"batter_name", "v3_prob", "savant_prob", "blend_prob"}
    if not needed.issubset(df.columns):
        st.info("Probability comparison chart needs batter_name, v3_prob, savant_prob, and blend_prob.")
        return

    chart_df = df.head(25).copy()

    for c in ["v3_prob", "savant_prob", "blend_prob"]:
        chart_df[c] = pd.to_numeric(chart_df[c], errors="coerce") * 100

    long_df = chart_df.melt(
        id_vars=["batter_name"],
        value_vars=["v3_prob", "savant_prob", "blend_prob"],
        var_name="model_signal",
        value_name="probability_pct",
    )

    st.bar_chart(
        long_df,
        x="batter_name",
        y="probability_pct",
        color="model_signal",
    )


def main() -> None:
    st.set_page_config(
        page_title="QEPC MLB Hit Board",
        page_icon="⚾",
        layout="wide",
    )

    st.title("⚾ QEPC-MLB Daily Batter 1+ Hit Board")
    st.caption("Production v4.1: 80% v3 + 20% Savant-light, with v3 top-5 protection gate.")

    with st.sidebar:
        st.header("Controls")

        selected_date = st.date_input("Score date", value=date.today())
        score_date = selected_date.isoformat()

        top_n = st.number_input("Top N", min_value=5, max_value=100, value=25, step=5)

        skip_fetch = st.checkbox(
            "Skip lineup fetch",
            value=False,
            help="Use the existing lineup CSV for this date instead of fetching from MLB API.",
        )

        if st.button("Run daily blend board", type="primary"):
            with st.spinner("Running QEPC daily blend pipeline..."):
                code, stdout, stderr = run_daily_pipeline(score_date, int(top_n), skip_fetch)

            if code == 0:
                st.success("Daily board completed.")
            else:
                st.error(f"Pipeline failed with exit code {code}.")

            with st.expander("Pipeline stdout", expanded=False):
                st.code(stdout or "(empty)")

            if stderr:
                with st.expander("Pipeline stderr", expanded=True):
                    st.code(stderr)

        st.divider()

        date_tag = date_tag_from_iso(score_date)
        top_csv = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_top{int(top_n)}_{date_tag}.csv"
        top_parquet = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_top{int(top_n)}_{date_tag}.parquet"
        summary_path = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_summary_{date_tag}.json"
        all_preds_path = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_all_predictions_{date_tag}.parquet"

        st.subheader("Selected files")
        st.text(f"Top board: {top_csv.relative_to(PROJECT_ROOT) if top_csv.exists() else 'not found'}")
        st.text(f"Summary: {summary_path.relative_to(PROJECT_ROOT) if summary_path.exists() else 'not found'}")

        if st.button("Use latest available board"):
            latest = latest_file("batter_1plus_hit_blend_top*.csv")
            if latest:
                st.session_state["override_top_csv"] = latest
                st.success(f"Loaded latest: {latest.name}")
            else:
                st.warning("No blend top board files found.")

    override_top_csv = st.session_state.get("override_top_csv")
    if override_top_csv:
        top_csv = Path(override_top_csv)
        # Try to infer matching summary from filename.
        possible_tag = top_csv.stem.split("_")[-1]
        summary_path = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_summary_{possible_tag}.json"
        all_preds_path = PROJECT_ROOT / PRED_DIR / f"batter_1plus_hit_blend_all_predictions_{possible_tag}.parquet"

    board = read_table(top_csv)
    if board is None:
        board = read_table(top_parquet)

    summary = load_json(summary_path)

    tab_board, tab_explorer, tab_model, tab_files = st.tabs(
        ["Daily Board", "Player Explorer", "Model Signals", "Files / Status"]
    )

    with tab_board:
        st.subheader("Daily Top Board")

        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows scored", f"{summary.get('rows_scored', '—')}")
            c2.metric("Top board avg", f"{summary.get('top_board_avg_blend_prob', 0) * 100:.1f}%" if summary.get("top_board_avg_blend_prob") is not None else "—")
            c3.metric("Max probability", f"{summary.get('top_board_max_blend_prob', 0) * 100:.1f}%" if summary.get("top_board_max_blend_prob") is not None else "—")
            c4.metric("Blend", f"{int(summary.get('v3_weight', 0.8) * 100)}/{int(summary.get('savant_weight', 0.2) * 100)}")

        if board is None:
            st.warning("No top board found for the selected date. Run the daily blend board from the sidebar.")
        else:
            board = format_board(board)

            teams = sorted([x for x in board.get("team", pd.Series(dtype=str)).dropna().unique()])
            selected_teams = st.multiselect("Filter teams", teams, default=[])

            search = st.text_input("Search player")

            view = board.copy()

            if selected_teams and "team" in view.columns:
                view = view[view["team"].isin(selected_teams)]

            if search and "batter_name" in view.columns:
                view = view[view["batter_name"].astype(str).str.contains(search, case=False, na=False)]

            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download displayed board as CSV",
                data=view.to_csv(index=False),
                file_name=f"qepc_hit_board_{score_date}.csv",
                mime="text/csv",
            )

            st.subheader("Top Board Probability Comparison")
            make_probability_chart(view)

    with tab_explorer:
        st.subheader("Player Explorer")

        all_preds = read_table(all_preds_path)

        if all_preds is None:
            st.info("No all-predictions parquet found for this date yet.")
        else:
            df = all_preds.copy()

            if "batter_name" not in df.columns:
                st.warning("All predictions file does not include batter_name.")
            else:
                player_names = sorted(df["batter_name"].dropna().astype(str).unique())
                player = st.selectbox("Choose player", player_names)

                p = df[df["batter_name"].astype(str).eq(player)].copy()

                if p.empty:
                    st.warning("No rows for selected player.")
                else:
                    show_cols = [
                        "batter_name",
                        "team",
                        "opponent",
                        "lineup_slot",
                        "v3_prob",
                        "savant_prob",
                        "blend_prob_base_80_20",
                        "v3_rank_daily",
                        "blend_prob",
                    ]
                    show_cols = [c for c in show_cols if c in p.columns]

                    st.dataframe(p[show_cols], use_container_width=True, hide_index=True)

                    cols = probability_cols(p)
                    if cols:
                        st.write("Signal values")
                        st.dataframe(p[cols], use_container_width=True, hide_index=True)

    with tab_model:
        st.subheader("Model Signals")

        if summary:
            st.json(
                {
                    "safety_version": summary.get("safety_version"),
                    "v3_model_dir": summary.get("v3_model_dir"),
                    "savant_model_dir": summary.get("savant_model_dir"),
                    "v3_weight": summary.get("v3_weight"),
                    "savant_weight": summary.get("savant_weight"),
                    "gate": summary.get("gate"),
                    "gate_rule": summary.get("gate_rule"),
                    "note": summary.get("note"),
                }
            )
        else:
            st.info("No summary JSON found yet.")

        if board is not None:
            st.subheader("Model disagreement")
            b = board.copy()

            if {"v3_prob", "savant_prob"}.issubset(b.columns):
                b["savant_minus_v3"] = pd.to_numeric(b["savant_prob"], errors="coerce") - pd.to_numeric(b["v3_prob"], errors="coerce")
                b["abs_disagreement"] = b["savant_minus_v3"].abs()

                cols = [
                    "batter_name",
                    "team",
                    "opponent",
                    "lineup_slot",
                    "v3_prob",
                    "savant_prob",
                    "savant_minus_v3",
                    "abs_disagreement",
                    "blend_prob",
                ]
                cols = [c for c in cols if c in b.columns]

                st.dataframe(
                    b.sort_values("abs_disagreement", ascending=False)[cols],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Need v3_prob and savant_prob columns to show disagreement.")

    with tab_files:
        st.subheader("File / Pipeline Status")

        status_rows = []

        paths = [
            PROJECT_ROOT / "artifacts/mlb/models/batter_1plus_hit_ranker_champion_v3/model.joblib",
            PROJECT_ROOT / "artifacts/mlb/models/batter_1plus_hit_ranker_savant_light_challenger/model.joblib",
            PROJECT_ROOT / "cache/mlb/processed/batter_savant_features_2021_2025.parquet",
            PROJECT_ROOT / "qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py",
            summary_path,
            top_csv,
            all_preds_path,
        ]

        for p in paths:
            status_rows.append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)) if p.exists() else str(p),
                    "exists": p.exists(),
                    "size_mb": round(p.stat().st_size / 1_000_000, 2) if p.exists() else None,
                }
            )

        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        st.subheader("Raw summary JSON")
        if summary:
            st.json(summary)
        else:
            st.info("No summary available.")


if __name__ == "__main__":
    main()
