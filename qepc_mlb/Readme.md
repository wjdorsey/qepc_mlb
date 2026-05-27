Yes — absolutely. We’ve changed enough that the README should become the “future Will doesn’t have to remember all this” document.

I’d update it with:

```text
current production model
daily command
dashboard command
major artifacts
what v4.1 means
what experiments were tested but not promoted
```

Run this from terminal:

````bash
cd ~/qepc_mlb_project

python - <<'PY'
from pathlib import Path
import re

path = Path("README.md")
text = path.read_text() if path.exists() else "# QEPC MLB\n"

section_title = "## Batter 1+ Hit Daily Board"

section = r'''
## Batter 1+ Hit Daily Board

Current production board:

```text
QEPC-MLB Daily Batter 1+ Hit Board v4.1
````

Production scoring logic:

```text
Base score:
  80% Batter 1+ Hit Ranker v3
  20% Batter Savant-light challenger

Gate:
  If v3 daily rank <= 5:
      use 90% v3 + 10% Savant-light
  Else:
      use 80% v3 + 20% Savant-light
```

The v4.1 board is the current recommended daily-use version.

### Run the daily board

For today's board:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --top_n 25
```

For a specific date:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --date YYYY-MM-DD \
  --top_n 25
```

Example historical test:

```bash
python qepc_mlb/predict/run_batter_1plus_hit_daily_blend.py \
  --date 2025-09-28 \
  --top_n 25
```

Shortcut script:

```bash
./scripts/run_today_hit_board_blend.sh
```

### Launch the Streamlit dashboard

```bash
streamlit run qepc_mlb/dashboard/app_batter_hit_board.py
```

Or:

```bash
./scripts/run_hit_dashboard.sh
```

The dashboard includes:

```text
Daily Board
Player Explorer
Model Signals
Files / Status
```

### Main output folder

```text
artifacts/mlb/predictions/batter_1plus_hit_blend/
```

Main daily output files:

```text
batter_1plus_hit_blend_top25_YYYYMMDD.csv
batter_1plus_hit_blend_top25_YYYYMMDD.parquet
batter_1plus_hit_blend_all_predictions_YYYYMMDD.parquet
batter_1plus_hit_blend_summary_YYYYMMDD.json
```

### Model artifacts

Primary v3 model:

```text
artifacts/mlb/models/batter_1plus_hit_ranker_champion_v3/
```

Savant-light challenger:

```text
artifacts/mlb/models/batter_1plus_hit_ranker_savant_light_challenger/
```

### Current production backtest notes

The clean game-id-safe monthly blend test showed:

```text
80% v3 / 20% Savant-light blend:
Top 5 daily hit rate:  ~74.19%
Top 10 daily hit rate: ~73.06%
Top 25 daily hit rate: ~70.67%
```

The v4.1 top-5 protection gate showed:

```text
Top 5 daily hit rate:  ~74.28%
Top 10 daily hit rate: ~73.06%
Top 25 daily hit rate: ~70.67%
```

This is why v4.1 is the current production daily board.

### Feature/model evolution

Production v3 includes:

```text
hitter rolling form
season-prior hitter stats
confirmed lineup slot
environment context
opposing starter prior-form context
batter-vs-pitcher handedness matchup context
```

Savant-light challenger adds prior-season batter contact-quality features:

```text
xBA / xwOBA style indicators
exit velocity
hard-hit indicators
barrel indicators
sweet-spot/contact quality
K/BB/whiff/chase percentiles
bat speed / squared-up rate where available
```

### Experiments tested but not promoted

These were useful, but are not the current production board:

```text
Full platoon split stack:
  added noise and did not beat v3 overall.

Platoon-light:
  did not beat v3.

Full opposing pitching staff context:
  too many features, did not beat v3 in scout testing.

Staff-light:
  helped some calibration/top-25 areas but hurt top-5/top-10.

Prior-season opposing-starter Savant:
  promising calibration signal, but sparse coverage.

Full Batter Savant model:
  promising, but not better than v3 across the full board.

Batter Savant Contact-Core:
  recovered top-5 somewhat, but lost top-10/top-25.

Meta-ranker / resolver:
  improved calibration in spots, but did not beat the simple blend.

Rule-based gated blend:
  v3 top-5 protection became v4.1 production logic.
```

### Important safety notes

```text
- The production daily board depends on confirmed MLB lineups.
- Run after lineups are posted.
- Rolling/history features are only as current as cache/mlb/processed/ data.
- Raw data, processed data, prediction outputs, and model artifacts are generally not committed to Git.
- Code, scripts, and documentation should be committed.
```

'''

pattern = rf"(?ms)^{re.escape(section_title)}\n.*?(?=^## |\Z)"

if re.search(pattern, text):
text = re.sub(pattern, section.strip() + "\n\n", text)
else:
text = text.rstrip() + "\n\n" + section.strip() + "\n"

path.write_text(text)
print("Updated README.md with v4.1 daily board + dashboard docs")
PY

````

Then commit it:

```bash
git status
git add README.md
git commit -m "Document daily batter hit board v4.1 and dashboard"
git push
````

That README update will make the repo way easier to pick back up later without needing to dig through chat archaeology.
