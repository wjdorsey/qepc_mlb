#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

streamlit run qepc_mlb/dashboard/app_batter_hit_board.py
