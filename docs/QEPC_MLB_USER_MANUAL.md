# QEPC-MLB User Manual

## Current Production System

Current production board:

```text
QEPC-MLB Daily Batter 1+ Hit Board v4.1
````

Production logic:

```text
Base:
  80% v3 Batter 1+ Hit Ranker
  20% Savant-light contact-quality challenger

Gate:
  If v3 daily rank <= 5:
      use 90% v3 + 10% Savant-light
  Else:
      use 80% v3 + 20% Savant-light
```

The goal of this board is to rank MLB hitters by estimated probability of recording **1+ hit** in a game.

---

## Quick Start

From the project root:

```bash
cd ~/qepc_mlb_project
```

Run today’s board:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --top_n 25
```

Run a specific date:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --date YYYY-MM-DD \
  --top_n 25
```

Example:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --date 2025-09-28 \
  --top_n 25
```

Shortcut:

```bash
./scripts/run_today_hit_board_blend.sh
```

---

## Dashboard

Launch the Streamlit dashboard:

```bash
./scripts/run_hit_dashboard.sh
```

Or directly:

```bash
streamlit run qepc_mlb/dashboard/app_batter_hit_board.py
```

Dashboard tabs:

```text
Daily Board:
  View the current top board, filter by team/player, download CSV.

Player Explorer:
  Inspect individual player model signals.

Results Checker:
  Fetch MLB API actuals and evaluate a board.

History:
  Track top 5 / top 10 / top 25 performance over time.

Model Signals:
  View model weights, gate logic, and disagreement information.

Files / Status:
  Confirm that required model/data files exist.
```

---

## Main Output Files

Daily prediction outputs are saved here:

```text
artifacts/mlb/predictions/batter_1plus_hit_blend/
```

Main files:

```text
batter_1plus_hit_blend_top25_YYYYMMDD.csv
batter_1plus_hit_blend_top25_YYYYMMDD.parquet
batter_1plus_hit_blend_all_predictions_YYYYMMDD.parquet
batter_1plus_hit_blend_summary_YYYYMMDD.json
```

The CSV file is usually the easiest file to open and review.

---

## Results Checker

The Results Checker can evaluate a saved board against actual results.

Preferred method:

```text
Use MLB API actuals
```

This fetches official MLB boxscore data and does not require local Retrosheet-style files to be updated.

Manual API actuals fetch:

```bash
python qepc_mlb/ingest/fetch_mlb_batter_actuals.py \
  --date YYYY-MM-DD
```

Evaluate a board using API actuals:

```bash
python qepc_mlb/evaluation/evaluate_daily_hit_board.py \
  --date YYYY-MM-DD \
  --top_n 25 \
  --actuals cache/mlb/results/mlb_batter_actuals_YYYYMMDD.parquet
```

Example:

```bash
python qepc_mlb/ingest/fetch_mlb_batter_actuals.py \
  --date 2025-09-28

python qepc_mlb/evaluation/evaluate_daily_hit_board.py \
  --date 2025-09-28 \
  --top_n 25 \
  --actuals cache/mlb/results/mlb_batter_actuals_20250928.parquet
```

---

## Evaluation History

Every successful evaluation updates:

```text
artifacts/mlb/evaluation/batter_1plus_hit_blend/evaluation_history.csv
```

This file tracks:

```text
date
top5 hit rate
top10 hit rate
top25 hit rate
top50 hit rate
missing actuals
merge keys
prediction file
actuals file
```

The dashboard History tab reads this file and displays rolling trends.

---

## Important Model Artifacts

Primary v3 model:

```text
artifacts/mlb/models/batter_1plus_hit_ranker_champion_v3/
```

Savant-light challenger:

```text
artifacts/mlb/models/batter_1plus_hit_ranker_savant_light_challenger/
```

Daily v4.1 blend runner:

```text
qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py
```

Dashboard:

```text
qepc_mlb/dashboard/app_batter_hit_board.py
```

Results evaluator:

```text
qepc_mlb/evaluation/evaluate_daily_hit_board.py
```

MLB API actuals fetcher:

```text
qepc_mlb/ingest/fetch_mlb_batter_actuals.py
```

---

## Data Notes

The daily board depends on confirmed MLB lineups.

Best practice:

```text
Run after confirmed lineups are posted.
```

If the board looks empty or incomplete, lineups may not be available yet.

The model also depends on local processed historical features. These live under:

```text
cache/mlb/processed/
```

Important processed files include:

```text
batter_game_logs.parquet
pitcher_game_logs.parquet
batter_game_logs_opp_pitcher_matchup.parquet
batter_game_logs_opp_pitcher_matchup_batter_savant_light.parquet
batter_savant_features_2021_2025.parquet
```

---

## Common Commands

Check Git status:

```bash
git status
```

Commit code/documentation changes:

```bash
git add .
git commit -m "Describe the change"
git push
```

Open prediction folder on Mac:

```bash
open artifacts/mlb/predictions/batter_1plus_hit_blend/
```

Open evaluation folder on Mac:

```bash
open artifacts/mlb/evaluation/batter_1plus_hit_blend/
```

---

## Troubleshooting

### The daily board has no rows

Likely cause:

```text
Confirmed lineups are not posted yet.
```

Try again closer to game time.

### Results Checker has missing actuals

Check whether MLB API actuals were fetched:

```bash
ls -lh cache/mlb/results/
```

Fetch actuals manually:

```bash
python qepc_mlb/ingest/fetch_mlb_batter_actuals.py \
  --date YYYY-MM-DD
```

Then rerun the evaluator.

### Dashboard will not launch

Make sure Streamlit is installed:

```bash
python -m pip install streamlit
```

Then run:

```bash
streamlit run qepc_mlb/dashboard/app_batter_hit_board.py
```

### The model files are missing

Check:

```bash
ls -lh artifacts/mlb/models/
```

Required folders:

```text
batter_1plus_hit_ranker_champion_v3
batter_1plus_hit_ranker_savant_light_challenger
```

---

## Production Baseline

As of the current checkpoint, the production system is:

```text
QEPC-MLB Daily Batter 1+ Hit Board v4.1
```

Do not overwrite this production path with experiments.

Experimental work should go into separate files or artifact folders, such as:

```text
artifacts/mlb/backtests/experimental_*
artifacts/mlb/models/experimental_*
```

Promotion rule:

```text
Only replace the production board if a challenger improves top-board performance
without creating worse calibration or reliability.
```

Primary board objectives:

```text
Top 5 hit rate
Top 10 hit rate
Top 25 hit rate
Consistency over forward-tracked days
```

---

## Recommended Daily Workflow

1. Run the daily board after lineups are confirmed.

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --top_n 25
```

2. Review the board in Streamlit.

```bash
./scripts/run_hit_dashboard.sh
```

3. After games are final, evaluate the board using MLB API actuals.

Use the dashboard Results Checker tab, or run:

```bash
python qepc_mlb/ingest/fetch_mlb_batter_actuals.py \
  --date YYYY-MM-DD

python qepc_mlb/evaluation/evaluate_daily_hit_board.py \
  --date YYYY-MM-DD \
  --top_n 25 \
  --actuals cache/mlb/results/mlb_batter_actuals_YYYYMMDD.parquet
```

4. Review the History tab for tracked performance.

---

## Current Philosophy

The daily board is now a usable production tool.

Future experiments should be tested against the production v4.1 baseline, not casually mixed into the production runner.

The current model philosophy is:

```text
Use v3 as the stable ranking engine.
Use Savant-light as a contact-quality signal.
Use simple, tested gates instead of overfitted complexity.
Track forward results before making major changes.
```

MD

````

Then add a README link to the manual:

```bash
python - <<'PY'
from pathlib import Path

path = Path("README.md")
text = path.read_text() if path.exists() else "# QEPC MLB\n"

line = "- [QEPC-MLB User Manual](docs/QEPC_MLB_USER_MANUAL.md)"

if line not in text:
    text = text.rstrip() + "\n\n## Documentation\n\n" + line + "\n"

path.write_text(text)
print("Linked user manual from README.md")
PY
````

Then commit it:

```bash
git status

git add README.md docs/QEPC_MLB_USER_MANUAL.md

git commit -m "Add QEPC MLB user manual"

git push
```

That gives us three layers now:

```text
README.md = repo overview
docs/QEPC_MLB_USER_MANUAL.md = how to operate the system
dashboard = interactive daily tool
```


