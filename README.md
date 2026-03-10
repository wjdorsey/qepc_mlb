# QEPC-MLB Architecture Plan

## Vision
QEPC-MLB should share the scientific, simulation-heavy spirit of the NBA project while being its own system. Baseball is not basketball with bats. The model should treat MLB as an event-driven environment built around pitcher-batter interactions, park effects, weather, lineup context, and bullpen state.

The core design principle is to separate:

1. **True talent** — what the player is underneath recent noise
2. **Current form** — what the player has been doing lately, using leakage-safe rolling features
3. **Matchup context** — handedness, pitch mix, park, weather, opponent quality, lineup slot
4. **Game environment** — run-scoring conditions, bullpen fatigue, umpire zone tendencies if added later
5. **Outcome simulation** — event-level or plate-appearance-level Monte Carlo generation

---

## Recommended Data Stack

### Tier 1 Core Sources

**Baseball Savant / Statcast**
- Best source for underlying player skill and contact quality
- Use for: xwOBA, xBA, xSLG, barrel%, hard-hit%, launch angle, sprint speed, pitch movement, velocity, whiff%, chase%, bat speed, squared-up%, etc.
- Role in QEPC-MLB: **talent engine**

**MLB Stats API**
- Best source for daily schedules, probable pitchers, game IDs, lineups/game feeds, team context
- Role in QEPC-MLB: **daily context engine**

**Retrosheet**
- Best source for long historical play-by-play and event files
- Use for: robust historical backtesting, sequence modeling, rest, bullpen usage reconstruction, lineup and game-state features
- Role in QEPC-MLB: **historical event engine**

**FanGraphs**
- Best source for park factors and useful contextual leaderboards/splits
- Role in QEPC-MLB: **environment calibration layer**

**Chadwick Register**
- Best source for ID crosswalks across MLBAM, FanGraphs, Retrosheet, Lahman, Baseball Reference
- Role in QEPC-MLB: **canonical identity layer**

**Weather source (example: Open-Meteo)**
- Use for hourly temperature, wind speed, wind direction, humidity, precipitation risk
- Role in QEPC-MLB: **game environment layer**

### Tier 2 Supporting Sources

**Lahman Database**
- Stable historical metadata and reference tables
- Useful for warehousing and fallback joins

**pybaseball / baseballr style wrappers**
- Useful ingestion layer, but not source of truth
- Good for accelerating prototype development

---

## QEPC-MLB Identity

This should not be framed as “QEPC NBA ported to MLB.”

Suggested framing:
- **QEPC-MLB: Quantum Event Cascade**
- In baseball, each plate appearance is a local collapse event inside a larger game wave.
- Pitchers create field pressure.
- Batters respond with contact, patience, aggression, or whiff behavior.
- Parks and weather distort the outcome field.
- Bullpens alter late-game state transitions.

That gives the project its own lore while still staying grounded in real predictive mechanics.

---

## Primary Model Targets

### Hitters (recommended first targets)
1. **Hits**
2. **Total Bases**
3. **Home Runs**
4. **Strikeouts**
5. **Walks**

### Pitchers (recommended first targets)
1. **Strikeouts**
2. **Earned Runs**
3. **Outs Recorded / Innings Pitched**
4. **Hits Allowed**
5. **Walks Allowed**

### Why these first?
These are cleaner event-native targets with better signal pathways than noisier stats like RBI or runs scored, which depend heavily on teammate sequencing.

---

## Feature Families

## 1) Batter True-Talent Features
- xwOBA
- xBA
- xSLG
- average EV
- max EV
- hard-hit%
- barrel%
- sweet-spot%
- K%
- BB%
- chase%
- whiff%
- contact%
- sprint speed
- bat speed
- squared-up%
- blasts / fast-swing rate
- handedness splits vs RHP/LHP

## 2) Pitcher True-Talent Features
- xERA
- FIP / SIERA if available
- xwOBA allowed
- xBA allowed
- xSLG allowed
- K%
- BB%
- whiff%
- chase%
- CSW%
- GB%
- FB%
- barrel% allowed
- hard-hit% allowed
- average EV allowed
- velocity by pitch type
- movement / release / extension
- pitch-mix composition
- handedness splits vs LHB/RHB

## 3) Leakage-Safe Rolling Form
For both hitters and pitchers, compute rolling and expanding windows with shift(1) only:
- last 3 / 5 / 10 / 20 games
- season-to-date prior to current game
- rolling PA-based rates
- rolling contact-quality metrics
- rolling pitch-arsenal quality
- rolling strikeout / walk form
- rolling rest / fatigue indicators

## 4) Matchup Interaction Features
These are the sauce.

### Batter vs Starter interaction
- batter split vs pitcher handedness
- batter weakness vs pitcher pitch mix
- pitcher vulnerability to batter batted-ball profile
- chase mismatch
- whiff mismatch
- GB/FB interaction
- platoon advantage

### Team lineup interaction
- projected lineup strength behind hitter
- lineup protection / surrounding OBP and SLG
- lineup strikeout tendency vs starter profile
- lineup patience vs starter command

## 5) Environment Features
- park factor overall
- handedness-specific park HR factor
- weather: temp, wind out/in, humidity
- roof open/closed if available
- travel / getaway game flags later if desired

## 6) Bullpen and Team Context
- bullpen workload previous 1/3 days
- bullpen quality trailing the starter
- expected leash for starter
- defensive support quality later if useful

---

## Modeling Philosophy

## Separate models by target
Do not force one monolithic model to predict everything.

Recommended first model set:
- hitter_hits_model
- hitter_total_bases_model
- hitter_hr_model
- pitcher_strikeouts_model
- pitcher_earned_runs_model
- pitcher_outs_model

## Baseline math by target
- **Strikeouts / walks / hits / HR**: Poisson or Negative Binomial style count models
- **Total bases**: count model or event-sim aggregation model
- **Earned runs**: overdispersed count model
- **Outs recorded**: bounded duration / count model with leash logic

## Calibration
Every target should have:
- raw model prediction
- shrinkage to stable priors
- post-hoc calibration layer
- uncertainty estimate / sigma band

---

## Monte Carlo Design

### Option A: direct stat simulation
For each target, simulate from calibrated count distributions.
Good for faster prototyping.

### Option B: plate-appearance simulation
For each batter PA vs pitcher context:
1. simulate PA outcome probabilities
2. aggregate plate appearances into game totals
3. transition to bullpen after starter exit logic

This is more accurate and more “QEPC-MLB,” but heavier to build.

### Recommendation
Start with Option A for v1, then move to plate-appearance simulation in v2.

---

## Suggested Project Structure

```text
qepc_mlb/
  __init__.py
  config.py
  ids/
    build_id_map.py
    player_id_crosswalk.parquet
  ingest/
    ingest_savant.py
    ingest_mlb_api.py
    ingest_retrosheet.py
    ingest_fangraphs.py
    ingest_weather.py
  features/
    build_batter_features.py
    build_pitcher_features.py
    build_matchup_features.py
    build_environment_features.py
  models/
    hitter_hits_model.py
    hitter_total_bases_model.py
    hitter_hr_model.py
    pitcher_strikeouts_model.py
    pitcher_earned_runs_model.py
    pitcher_outs_model.py
  simulation/
    simulate_game_props.py
    simulate_plate_appearances.py
    bullpen_transition.py
  backtests/
    walk_forward_hitter_hits.py
    walk_forward_pitcher_ks.py
    calibration_report.py
  utils/
    schema.py
    time.py
    park_factors.py
    weather.py
    joins.py
```

### Data cache layout
```text
cache/mlb/
  raw/
    savant/
    mlb_api/
    retrosheet/
    fangraphs/
    weather/
  processed/
    player_game_logs.parquet
    pitcher_game_logs.parquet
    batter_features.parquet
    pitcher_features.parquet
    matchup_features.parquet
    daily_projection_input.parquet
  ids/
    player_crosswalk.parquet
```

---

## Build Order

## Phase 1 — data spine
1. Build canonical player/team/game ID layer
2. Ingest Savant leaderboard and Statcast extracts
3. Ingest MLB schedule + probable pitchers
4. Ingest Retrosheet game/event history
5. Build unified player-game tables

## Phase 2 — first modelable tables
1. Batter game-log feature table
2. Pitcher game-log feature table
3. Daily matchup table (starter vs projected hitters)
4. Park/weather join

## Phase 3 — first models
1. Pitcher strikeouts
2. Hitter strikeouts
3. Hitter total bases
4. Pitcher earned runs

## Phase 4 — simulator
1. distribution calibration
2. Monte Carlo totals
3. starter-to-bullpen transition logic
4. uncertainty bands and confidence diagnostics

---

## First Version Recommendation

For the very first serious QEPC-MLB prototype, focus on:
- **Pitcher strikeouts**
- **Hitter strikeouts**
- **Hitter total bases**

Why these?
- High signal
- Reasonable relationship to Savant skill metrics
- Strong dependence on handedness and pitch mix
- Easier to validate with game logs and lineups

---

## What to Avoid Early
- RBI and runs scored as primary targets
- overly fancy sequence simulation before clean data plumbing exists
- relying only on season leaderboard exports
- combining sources before ID reconciliation is stable
- building one “god model” for all outcomes

---

## Accuracy Priorities

Ranked by likely impact:
1. canonical ID and data-join quality
2. probable starter accuracy
3. handedness and pitch-mix matchup features
4. park/weather adjustments
5. leakage-safe rolling form
6. bullpen transition logic
7. post-hoc calibration by target

---

## Recommended Next Build

The next concrete coding step should be:

1. **build_qepc_mlb_data_spine.py**
   - unify IDs
   - ingest core sources
   - output batter/pitcher player-game tables

2. **build_qepc_mlb_matchups.py**
   - produce daily matchup rows for hitters and starters
   - attach park/weather/context

3. **qepc_mlb_pitcher_k_model.py**
   - first full target model
   - easiest strong entry point for predictive accuracy

That gives the project a clean beginning with its own identity and a path to serious walk-forward evaluation.

