#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python qepc_mlb/predict/run_batter_1plus_hit_daily.py \
  --top_n 25

echo ""
echo "Done. Check:"
echo "artifacts/mlb/predictions/batter_1plus_hit_ranker/"
