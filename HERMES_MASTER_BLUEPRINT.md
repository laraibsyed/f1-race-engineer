# HERMES — Master Blueprint (hand this whole file to the next Claude)

> **Goal of the next chat:** build ONE runnable file (`system/HERMES/hermes_master.py`) that joins every finished module to the Gate Tree (Tier 1→2→3) and the Execution Tree, so a single command produces a per-lap, per-driver pit/pace recommendation with explanations.
> Everything below was extracted from the actual code on 2026-09-26. Where something is an *assumption*, a *bug*, or *missing*, it says so explicitly. Nothing here is guessed. Companion file: `HERMES_CODE_BUNDLE.md` (verbatim source of the core modules).
> **Not in the repo, so I can't describe them — attach if you have them:** the MDP document (paragraph numbers are cited throughout the code, e.g. "para 114-116"), the Gate Tree diagram, the Execution Tree diagram, the "Second Driver Problem" diagram (Layers 1-3), `Master_Checklist.pdf`.

---

## 0. TL;DR for the builder

1. All modules work **standalone**. **None of them import each other.** The four tree files only import `dataclasses`/`typing`/`numpy`. The coupling is documented only in field comments ("from tyre_life_projection", "EXTERNAL – live feed").
2. **Most filenames contain hyphens** (`gate-tier-3.py`, `execution-tree.py`, `sc-gamble.py`, `knowledge.py` is fine). They cannot be `import`ed normally → load with `importlib.util.spec_from_file_location`. Their `if __name__ == "__main__":` blocks are guarded, so loading is side-effect free (except `standings.py`, which `os.makedirs` at import, and modules that `import google.cloud.storage` at top level).
3. **Of the 3 Tier-3 tyre inputs, all 3 exist. Of the 5 Tier-3 race-context booleans, 0 exist as live functions** (`undercut_opportunity`, `overcut_opportunity`, `rival_undercut_threat`, `in_dirty_air`, plus `safety_car_deployed` which is trivial from `TrackStatus`). The rival module is a *historical batch detector that uses hindsight* (see §7-B4) and must be re-expressed as a live rule.
4. **There is a confirmed silent bug that makes `track_temp_bucket` a no-op in the tyre projection** (§7-B1). Fix before trusting any projection.
5. There is **no code** for: `get_driver_stress_trigger()`, `expected_stint_length`, `can_delay_one_lap_without_position_loss`, Second-Driver `priority_driver_id`, a per-lap state builder, or an orchestrator. Those are the actual work.

---

## 1. Project context

* **What it is:** an F1 race-engineer decision system ("HERMES") for a *two-car team* (D1 / D2). Historical data 2018–2026 (2026 partial). Dissertation-stage project: the author repeatedly insists on **"validate on one race, then scale"**, **"never guess a schema — inspect real data first"**, **"flag assumptions explicitly (`# ASSUMPTION`)"**, **"return `None`, never a guess"**.
* **Decision framework:** an MDP. Action space (from code comments): pit-related (`Pit Both / Pit One / Stay Out`), team-related (`Driver 1 Priority` = the default), pace (`Push / Neutral / Conserve / Manage Degradation`).
* **Reward (strategic-sacrifice module):** `R_team = α·ΔWCC + (1−α)·[w1·L1·ΔWDC1 + w2·L2·ΔWDC2] − λ·risk`.
* **Two-tier constraint philosophy (MDP):** Tier 1 = hard safety (force PIT_NOW, no reward calc). Tier 2 = regulatory feasibility (make actions illegal *before* reward). Tier 3 = soft constraints (modify reward / filter actions).
* **Team focus:** early tyre-model work trained/evaluated on Red Bull (`RBR_ALIASES`) then broadened to *all teams* for statistical power (`is_rbr` kept as a robustness flag).
* **Environment:** Windows 11, Python venv at `f1-race-engineer/.venv` (Python 3.14 mentioned in a comment), PowerShell/Git-Bash. Libraries actually used: pandas, numpy, scikit-learn, lifelines (Cox/KM), matplotlib, fastf1, google-cloud-storage, python-dotenv, requests, openpyxl (read_excel), scipy/statsmodels (VIF).
* **Git state:** the whole `f1-race-engineer/` folder is currently **untracked** (only "Initial commit" + "tyre usage information" exist at repo root).
* **Secrets (do NOT paste anywhere):** `f1-race-engineer/.env` (contains `GOOGLE_APPLICATION_CREDENTIALS`, `GITHUB_TOKEN`), `service-account.json`. Both are in `.gitignore`.

### Conventions used in every script (mirror them)
| Convention | Detail |
|---|---|
| `CachedBucket` | Read-through GCS wrapper, copy-pasted into ~30 scripts: `read_csv(blob_path)`, `list_blob_names(prefix)`, (`read_text`, `exists`, `write_csv` in knowledge.py). Local cache `./gcs_cache/<blob path>`. Bucket `f1-race-engineer-bucket` (env `BUCKET_NAME`); cache dir env `GCS_CACHE_DIR`. |
| "Duplicate deliberately" | Constants/functions are copy-pasted between files so each is standalone (e.g. `detect_cliff` lives in `cliff-detection.py`, `gate-tier-1.py`; `CLIFF_PROBABILITY_THRESHOLD=0.017` in `gate-tier-3.py` and `execution-tree.py`). Comments say "MUST stay in sync". **Decide in the orchestrator: import from one source and assert equality.** |
| Naming | Race folder = underscore style `Abu_Dhabi_Grand_Prix`; raw tracinginsights = spaces `Abu Dhabi Grand Prix`. Driver = 3-letter FastF1 code (`VER`). `DriverAhead` in telemetry is a car **number** string (or literal `"None"`). Session codes `R`, `S` (sprint), `FP1..`, `Q`. |
| Failure semantics | Unknown → `None`/`INSUFFICIENT_DATA`, never a fabricated number. Trees are None-safe (a `None` trigger input = trigger False). |
| Checkpointing | Expensive jobs checkpoint per race to `checkpoints/…` and resume. |

---

## 2. Data layout

### 2.1 GCS bucket `f1-race-engineer-bucket` (mirrored under `./gcs_cache/`)
```
raw/fastf1/<year>/<Race_Name>/<session>/{laps,messages,results,weather}.csv
clean/fastf1/<year>/<Race_Name>/<session>/laps_flagged.csv     # cleaning flags added
clean/features/<year>/<Race_Name>/<session>/laps_features.csv  # 889 files cached locally; THE main modelling table
raw/tracinginsights/<year>/<Race Name>/<Race|Sprint>/<DRV>/<lap>_tel.json   # ~197,909 files
clean/tracinginsights/<year>/<Race_Name>/<session>/telemetry_by_lap.csv     # only ~200 races
processed/driver_taxonomy_sampled_scores.csv
clean/fastf1/<y>/<race>/<session>/weather_cleaned.csv           # from weather/cleaning.py
```
### 2.2 Local files (repo, all under `f1-race-engineer/`)
| Path | What |
|---|---|
| `tyre_life_models.pkl` | **Production tyre models**: `{"reg_models": {(compound,circuit,era): LinearRegression}, "cph": CoxPHFitter, "temp_dummy_columns": ["temp_extreme","temp_hot","temp_warm"], "era_dummy_columns": ["regulation_era_2022-2025","regulation_era_2026+"]}`. 211 regression models (compounds HARD/HYPERSOFT/MEDIUM/SOFT/SUPERSOFT/ULTRASOFT; eras 2018-2021/2022-2025/2026+). Cox: stratified by `compound`, params `[temp_extreme,temp_hot,temp_warm,regulation_era_2022-2025,regulation_era_2026+,fuel_load_estimate,stint_number,circuit_degredation_ordinal]`, **concordance 0.619 (weak)**. |
| `cliff_detection_stints.csv` | One row per stint: `season,race,session,driver,stint,team,is_rbr,compound,circuit,n_laps_cleaned,n_laps_true,duration,event,cliff_tyre_age,slope_ratio,improvement,track_temp_bucket,fuel_load_estimate,stint_number,regulation_era`. Feeds Cox + threshold calibration. |
| `sc_vsc_circuit_level_prior.csv` | **The validated SC prior**: `circuit,n_races,p_race_incidence,mean_total_laps,p_window_horizon` (5-lap window; e.g. Baku 0.098, Abu Dhabi 0.044). Only circuits with ≥4 races. |
| `sc_vsc_empirical_prior.csv` | Lap-level diagnostic prior. **Do not use in decisions** (failed validation). |
| `driver_taxonomy_master_final.csv` | 39 drivers × `aggression_level, defensive_strength, tyre_management, consistency_factor, wet_weather_skill, pressure_risk_tolerance` (scaled 0.85–1.15, 1.0=average). |
| `driver_taxonomy_complete_roster.csv` | Above + `source` (`real_computed` or archetype fallback), `archetype`, `career_races` (44 drivers; <20 career races → archetype). |
| `driver_rolling_profiles.csv` | Leakage-safe point-in-time `tyre_management_pti`, `consistency_factor_pti` per (driver, season, round/target_race), `prior_career_races`, `profile_source` (`rolling`/`fallback`). |
| `laps_features_with_driver_profiles.csv` | 115 MB. `laps_features` + the 3 PTI columns. Used only by the driver ablation. |
| `checkpoints/rival_knowledge/…` | Per-race `*_opponents.csv` (178) / `*_opponents_enriched.csv` (173) / undercut windows+events; `archive_event_summary_enriched.csv` (cols `Driver,window_start_lap,window_end_lap,actual_pit_lap,direction,year,race,session,pit_loss_constant_s,rival,points_gap,risk_appetite`), `archive_per_race_analysis.csv` (**per-circuit empirical pit-loss constants**), `archive_run_log.csv`, `standings_cache/`. |
| `data/external/team-radios/classified_radios.csv` | 17,619 radio messages: `id,driver_id,racing_number,grand_prix,race_id,session_date,message_timestamp,transcription,category,stress_level`. Categories: other 6464, acknowledgement 2606, tyre_feedback 2349, position_gap_info 2178, pace_management 1654, mechanical_issue 1106, pit_instruction 832, safety_car_vsc 386, team_order 44. `stress_level` only on tyre_feedback: informational 1978 / medium 334 / high 37. |
| `src/config/f1_tyre_constraints.json` (also `../f1_tyre_constraints.json`) | Per-year `mandatory_dry_compounds, total_sets_allocated, wet_race_exception, q2_start_tyre_rule, source_article`. **Contradicts code in places — §7-B9.** |
| `src/config/f1_soft_triggers.json` | Keys: `parc_ferme, flags, pit_execution, driver_conduct, team_orders, penalties, race_status` (blue/yellow/double-yellow/red/VSC/SC rules with `mdp_action` strings). |
| `src/config/f1_year_changes.json` | Per-year 2018–2026 regulation diffs. |
| `src/taxanomy/circuit_taxonomy.xlsx` | 28 circuits: `circuit_id, circuit_name, circuit_type(Street/Track), circuit_degredation(low/medium/high), overtaking_difficulty(1-5), track_pos_sensitivity, double_stack_risk, unsafe_release_risk, sc_probability_pct, vsc_probability_pct, drs_zones_count, active_aero_zones, weather_volatility`. (**"degredation" is misspelled everywhere — keep the spelling.**) Manually estimated. |
| `src/taxanomy/drivers_archetypes.xlsx` | 4 fallback archetypes (`rookie_aggressive, rookie_conservative, junior_high_potential, senior_backmarker`) × 6 metrics. |
| `src/taxanomy/pirelli_tyres_reference.xlsx` | C3/C4/C5/Inter/Wet: `base_lap_time_delta, degradation_per_lap_seconds, optimal_temp_min/max, wet_performance_factor, lap_survival` (45/35/25/45/60). Unused so far — a candidate source for `expected_stint_length`. |
| `data_dictionary.csv` | Column profile of raw fastf1 datasets. |
| Result CSV/PNGs | `deadline_buffer_*`, `stint_length_sensitivity.csv`, `z_threshold_sensitivity.csv`, `sc_probability_*_validation.csv`, `ablation_*.csv`, `stress_signal_lookahead_sweep.csv`, `tyre_regression_v1/v2_results.csv`, `cliff_probability_roc.png`, EDA PNGs. |

### 2.3 `laps_features.csv` schema (54 cols in cache; +6 in joined file) — one row per (driver, lap)
`Time, Driver, DriverNumber, LapTime(timedelta str), LapNumber, Stint, PitOutTime, PitInTime, Sector{1,2,3}Time, Sector{1,2,3}SessionTime, SpeedI1, SpeedI2, SpeedFL, SpeedST, IsPersonalBest, Compound, TyreLife, FreshTyre, Team, LapStartTime, LapStartDate(all null), TrackStatus(str, e.g. "1","12","4","5"), Position(populated in R), Deleted, DeletedReason, FastF1Generated, IsAccurate,`
**flags:** `is_pit_in, is_pit_out, is_sc_lap, is_vsc_lap, is_missing_laptime, is_out_lap, is_in_lap, is_outlier_laptime`
**engineered:** `source_file, tyre_age(=TyreLife), compound_encoded(WET0,INTER1,HARD2,MED3,SOFT/HYPER/SUPER/ULTRA 4), stint_number(=Stint), gap_to_leader, gap_to_car_ahead (lap-based proxy from cumulative Time, NOT same-instant), degradation_rate((laptime−stint-best-clean)/tyre_age), fuel_load_estimate (=110kg·(1−(lap−1)/total_laps), linear, an assumption), track_temp_c, track_temp_bucket (cool ≤25 <warm ≤35 <hot ≤45 <extreme), is_sc_deployed_lap, is_sc_ending_lap, is_vsc_deployed_lap, is_vsc_ending_lap, is_sc_through_pit_lane_lap` (deployment = event *moment*; `is_sc_lap` = every lap under SC), `Season, Race`; joined file adds `prior_career_races, tyre_management_pti, consistency_factor_pti, profile_source`.
TrackStatus codes: `"1"` clear, `"2"` yellow, `"4"` SC, `"5"` red, `"6"/"7"` VSC (string may concatenate several codes within a lap). Red flag detection in code = `TrackStatus` contains `"5"`.

### 2.4 Raw telemetry JSON (`<lap>_tel.json`)
`{"tel": {time, rpm, speed, gear, throttle, brake, drs, distance, rel_distance, DriverAhead, DistanceToDriverAhead, acc_x/y/z, x, y, z, dataKey}}` — dict of parallel arrays. `DistanceToDriverAhead` is **metres** (not seconds). `laptimes.json` has FastF1-style short keys plus weather `wT,wAT,wH,wP,wR,wTT,wWD,wWS`.

---

## 3. Architecture (as implemented)

```
                    ┌──────────────── per lap, per driver ────────────────┐
 tyre models ──► tyre_life_projection.build_tyre_life_projection()
   (pkl)             └─► predicted_pace_loss (s)  +  cliff_probability_next_5_laps
                                   │
 TIER 1 (hard)   red_flag | tyre_damaged | cliff_already_hit(live detect_cliff) | unsafe_weather ─► PIT_NOW / MOVE_TO_TIER_2
 TIER 2 (rules)  mandatory 2-dry-compound rule (+deadline buffer 5 laps)             ─► PIT_FLEXIBLE / MOVE_TO_TIER_3
 TIER 3 (soft)   8 boolean triggers → count: 3+ = PIT_NOW, 1-2 = PIT_LATER, 0 = DONT_PIT
                 tyre triggers suppressed if SC-gamble says WAIT ; returns {decision, reason, triggers, driver_stress_signal}
 EXECUTION TREE  fires on PIT_NOW / PIT_FLEXIBLE: who pits (D1/D2/BOTH) → priority → avoid stacking? → double-stack/SC logic
                 + driving_instruction (PIT_LAP / MANAGE / NEUTRAL / PUSH / CONSERVE)
 LOOP            tyre_age+1 → re-run projection → back into Tier 1 and Tier 3 next lap
```
Design principles already fixed by the author (keep them):
* Tier 3 decides **WHETHER** to pit from objective signals; Execution decides **HOW** to drive. Execution uses `tier3_reason` and `driver_stress_signal` as *context only*, never re-evaluates the SC gamble or stress.
* `driver_stress_signal` is **not a vote** in Tier 3 (validation found no significant link between radio stress and pit timing).
* SC gamble is an **expected-cost comparison**, not a boolean; only its `WAIT` output feeds Tier 3, where it *suppresses tyre-only triggers* (`cliff_proximity, pace_lap_delta, tyre_age`), never external ones (undercut, active SC, rival threat, dirty air).
* Weather modules output a **strategic state**, never a pit call ("HERMES combines it with everything else").

---

## 4. Module catalogue (every module, its API, artifacts, provenance)

Paths are relative to `f1-race-engineer/system/HERMES/` unless noted.

### 4.1 `trees/` — the decision trees (PRIMARY INTEGRATION TARGET)

**`gate-tier-1.py`**
* `Tier1State(red_flag_or_race_stopped:bool, tyre_structurally_damaged:bool, unsafe_weather:bool, tyre_age_history:list, laptime_seconds_history:list)`; `evaluate_tier1(state)->"PIT_NOW"|"MOVE_TO_TIER_2"`. Order: red flag → damage → cliff-already-hit → weather.
* `detect_cliff(x,y)` piecewise-linear breakpoint scan; constants `MIN_SEGMENT_LENGTH=3, MIN_IMPROVEMENT=0.20, MIN_SLOPE_RATIO=2.0, MIN_STEP_SECONDS=0.3`; needs ≥6 laps else returns "not a cliff". `tyre_cliff_already_hit(ages, laptimes)->bool`.
* **Live-feed caveat:** the history lists must be *clean* laps of the **current stint** (exclude in/out laps, SC/VSC laps, outliers — same filter as cliff-detection) or SC-lap slowdowns will fake a cliff.

**`gate-tier-2.py`**
* `Tier2State(season, circuit, is_sprint_weekend, compound_history_dry:set, wet_race_exception:bool, laps_remaining_in_race:int, remaining_sets:dict, sets_used_so_far:int, stops_made_so_far:int)`; `evaluate_tier2->"MOVE_TO_TIER_3"|"PIT_FLEXIBLE"`.
* Actual logic: if mandatory 2-dry-compound rule done (or wet exception) → Tier 3; else if `laps_remaining <= DEADLINE_BUFFER_LAPS(5)` → `PIT_FLEXIBLE`; else Tier 3. **`remaining_sets`, `sets_used`, `stops_made`, `season`, `circuit` are not used by `evaluate_tier2`** (only by helper rules).
* Helpers implemented but unwired: `get_regulation_era(season)` (≤2021→"2018-2021", ≤2025→"2022-2025", else "2026+" — **single source of truth for era strings**), `check_degradation_model_compatible`, `get_dry_set_allocation` (13 std / 12 sprint for 2020+, `None` for 2018/19 → must come from FastF1), `apply_sprint_shootout_consumption` (2024+: −1 MEDIUM, −1 SOFT), `compound_available`, `q2_rule_applies` (2018-21, not 2021 sprint), `get_mandatory_start_compound`, Monaco two-stop rule (2025 only: ≥3 sets & ≥2 stops), `mandatory_compound_done`, `mandatory_compound_reset_on_red_flag()->False` (red flag never resets compound history).
* `compound_history_dry` must exclude INTERMEDIATE/WET.

**`gate-tier-3.py`** (calibrated thresholds — see §9)
* `Tier3State(cliff_probability_next_5_laps:Optional[float], predicted_pace_loss:Optional[float], tyre_age:float, expected_stint_length:float, undercut_opportunity, overcut_opportunity, safety_car_deployed, rival_undercut_threat, in_dirty_air: bool, driver_stress_signal:bool=False, sc_gamble_recommendation:Optional[str]=None)`.
* Triggers: `cliff_proximity = p >= 0.017`; `pace_lap_delta = loss >= 1.805 s`; `tyre_age = tyre_age >= 0.421 × expected_stint_length` (False if stint length ≤0); plus the 5 booleans. 3+ → `PIT_NOW`, 1-2 → `PIT_LATER`, 0 → `DONT_PIT`.
* `evaluate_tier3(state)->{"decision","reason"("SC_GAMBLE"|"SOFT_TRIGGERS"|None),"triggers":{8 names},"driver_stress_signal"}`. `reason="SC_GAMBLE"` only when `sc_gamble_recommendation=="WAIT"` *and* at least one tyre trigger was actually suppressed.

**`execution-tree.py`**
* `DriverPitContext(driver_id:"D1"|"D2", gate_tree_trigger_tier:int(1|2|3, lower=more urgent), tyre_age, track_position:int(1=lead), can_delay_one_lap_without_position_loss:bool[EXTERNAL], cliff_probability_next_5_laps, tier3_reason, driver_stress_signal)`; `ExecutionTreeState(triggered_drivers:list[1-2], safety_car_active, circuit, priority_driver_id=None)`; `evaluate_execution_tree(state)->{driver_id: {"decision","penalty_seconds","reason"?,"driving_instruction"}}`.
* Flow: 1 driver → `PIT_NOW`. 2 drivers → `resolve_priority` (Second-Driver override → lower tier number → D1 default) → if `other.can_delay…` → other `PIT_LATER` else `evaluate_double_stack`: SC active ⇒ the driver **further back** pits first, 2nd car takes `DOUBLE_STACK_PENALTY_SECONDS=5.0`; no SC ⇒ stagger by 1 lap (more urgent tier first).
* `get_driving_instruction(decision, cliff_prob, tier3_reason, driver_stress_signal)`: `PIT_NOW→PIT_LAP`; `SC_GAMBLE→MANAGE`; stress→`CONSERVE`; unknown→`MANAGE`; ratio=p/0.017: ≥1 `MANAGE`, ≥0.5 `NEUTRAL`, else `PUSH`.
* `get_pit_lane_speed_limit(circuit)` 80 km/h, 60 at `Monaco_Grand_Prix`.
* **Quirks:** (a) `get_driving_instruction` is defined **twice** (line 55 is a docstring-only stub; line 61 wins) — delete the stub. (b) The tree only outputs for *triggered* drivers; a non-triggered driver (Tier-3 `PIT_LATER`/`DONT_PIT`) gets no instruction — orchestrator must call `get_driving_instruction("PIT_LATER"/"DONT_PIT", …)` itself. (c) `PIT_FLEXIBLE` is treated identically to `PIT_NOW` (single-trigger → `PIT_NOW`) — semantics ("flexible") are lost.

**`tyre_life_projection.py`** — `build_tyre_life_projection(*, reg_models, cph, temp_dummy_columns, era_dummy_columns, compound, circuit, regulation_era, tyre_age, fuel_load_estimate, stint_number, track_temp_bucket, circuit_degredation_ordinal, cliff_horizon_laps=5) -> {"predicted_pace_loss": float|None, "cliff_probability_next_5_laps": float|None}`. Regression looks up `(compound,circuit,era)`; Cox gives `1 − S(t+5)/S(t)`. `None` when no model/stratum (WET/INTERMEDIATE always `None`). **Has the temp-bucket bug (§7-B1).** The docstring's "`fit_all_models()`" no longer exists — refit is `model-fit.py`, load the pickle.

**`model-fit.py`** — one-shot fit-and-pickle (`--cliff-stints-csv`, `--output tyre_life_models.pkl`). Reads GCS `clean/features/`, applies structural cleaning, fits Regression V2 per `(Compound, Race, regulation_era)` (`LinearRegression`, `MIN_ROWS_PER_GROUP=40`, target `pace_loss_seconds` = laptime − stint-best clean lap, features `tyre_age, fuel_load_estimate, stint_number` + temp dummies) and a Cox (penalizer 0.1, strata `compound`, WET excluded, `MIN_EVENTS_PER_COMPOUND=15`, race sessions only). Needs `src/taxanomy/circuit_taxonomy.xlsx` for the degradation ordinal (`low0/medium1/high2`) and `CIRCUIT_ID_TO_RACE_NAMES`. Fit on 100% of data (production model, not a validation run). Cleaning constants: `MIN_STINT_LENGTH=5`, `RED_FLAG_RESTART_BUFFER=2 laps`, `LAPTIME_OUTLIER_Z_THRESH=4.0` (robust MAD z).

**`threshold-calibration.py`** — produced the three Tier-3 thresholds (§9). **`sensitivity-buffer.py`** — produced `DEADLINE_BUFFER_LAPS` evidence (`deadline_buffer_sensitivity.csv`: 5 laps flags 0.41 % of historically compliant driver-races; 1 lap 0.27 %; 10 laps 1.5 %). **`test.py`** — loads the pickle and calls the projection (Bahrain MEDIUM age 15 ⇒ pace_loss 1.246 s, cliff p 0.0138; WET ⇒ both `None`).

### 4.2 `tyre/` — model fitting (offline)
`tyre-model-v1.py` (RBR-only, per compound+circuit, target `degradation_rate`; flagged circular), `tyre-regression-v2.py` (target `pace_loss_seconds`; stratified compound×race×era; three splits: random, fixed 2024 cutoff, expanding window; VIF check), `cliff-detection.py` (piecewise scan over each stint → `cliff_detection_stints.csv`; **true stint length `n_laps_true` computed from raw tyre_age before cleaning** — fixed an earlier undercount bug), `survival-analysis-v1.py` (Kaplan-Meier, compounds differ, log-rank p<0.0001), `survival-analysis-v2.py` (Cox, stratified by compound, WET excluded, sprints excluded, `is_rbr` robustness variant).
Key facts: dry compounds only in regression; WET/INTERMEDIATE follow drying physics not wear and have **no** tyre model. Cliff ROC (533 events): AUC 0.683. Cox concordance 0.619.

### 4.3 `safety-car/`
* `sc-vsc-probability-model.py`: `build_circuit_level_prior(laps)`, `get_sc_probability(circuit_prior_df, circuit, lap_number=None)->float|None` (**circuit-level constant; `lap_number` ignored by design**). Formula: `p_window = p_race_incidence × (horizon / mean_total_laps)` (Bernoulli + uniform placement; an earlier compound-hazard formula was rejected because it returns 100 % at p_race=1). Horizon 5 laps; min 4 races. To use live: read `sc_vsc_circuit_level_prior.csv` into a DataFrame (avoid the GCS import).
* `sc-vsc-probability-model-v2.py` / `sc-validation.py`: leave-one-race-out validation of the lap-level prior: **AUC 0.503, bootstrap 95 % CI [0.475, 0.533], worse Brier than base rate → lap-level rejected**.
* `sc-gamble.py`: `SCGambleInputs(p_sc_next_n_laps, predicted_pace_loss_per_lap, cliff_probability_next_n_laps, n_laps_horizon)`; `evaluate_sc_gamble->{"recommendation":"WAIT"|"NO_ADVANTAGE_TO_WAITING"|"INSUFFICIENT_DATA","cost_pit_now","cost_wait","expected_saving_if_wait"}`. Cost model: `pit_now = 22.0`; `if_sc = 11.0 + 1.5 + degr/2`; `if_no_sc = 22.0 + degr + 5.0×cliff_p`; `degr = pace_loss_per_lap × horizon`; `cost_wait = p·if_sc + (1−p)·if_no_sc`. **All cost constants are ASSUMPTIONS.** `compute_historical_sc_duration(laps)` is real data (mean SC length per circuit) but unused by the evaluator. See §7-B5 for input-semantics mismatches.

### 4.4 `weather/`
* `crossover.py` (slicks→inters): `evaluate_crossover(rain_probability_0_100, laps_remaining, total_laps, thresholds=None)->CrossoverResult(state, stage, …)`; states `STAY_SLICKS, MONITOR, CONSIDER_INTERS, BOX_INTERS(unreachable)`; stage by % remaining (>60 % early, 20–60 % mid, <20 % late); thresholds `(monitor, consider)`: early (45,60), mid (30,45), late (20,35); perturbation sets `baseline/stricter/looser`. **Provisional, not empirically calibrated.**
* `drying_line.py` (wets→slicks): `evaluate_drying_crossover(seconds_since_rain_end, switched_driver_lap:LapTimeSample|None, reference_laps:list|None, …)->DryingCrossoverResult(state: STAY_WET|GATE_OPEN_NO_EVIDENCE|CONSIDER_DRIER_TYRE, …)`. Stage 1 gate `MIN_SECONDS_SINCE_RAIN_STOPPED=300`; stage 2 evidence: switched car ≥1.0 s/lap faster than median of cars still on the wetter compound (caller must pre-filter in/out/SC laps).
* `openmeteo.py`: `get_historical_weather(race,date)`, `get_forecast_weather(race,days=16)` (free, no key) + `CIRCUIT_COORDS`; forecast `precipitation_probability` is the live rain input. `backtest.py`/`drying-backtest.py`/`sensitivity.py`/`threshold.py` = validation (Sochi 2021 showed archive/reanalysis rain is too coarse; a cost-derived crossover `P* = (C_dry_on_inters + C_pit)/(G_wet + C_wet_on_slicks + C_dry_on_inters)`), `cleaning.py` → `weather_cleaned.csv` (has `is_rain_end`).
* **Not connected to Tier 1 `unsafe_weather`, and there is no tyre model for wet compounds.**

### 4.5 `rival-awareness/`
* `knowledge.py` (the core; CLI per race): `build_opponent_table` (per-lap `ahead_driver, distance_to_ahead_m, ahead_compound, ahead_tyre_age` from raw telemetry), `add_behind_columns` (reverse lookup → `behind_driver, distance_to_behind_m, behind_compound, behind_tyre_age`), `compute_pit_lane_loss` (real `PitOutTime−PitInTime`, green-flag only, winsorised median, >120 s dropped), `flag_undercut_windows(opponent_df, features_df, pit_loss_s, window_laps=3, min_tyre_age_gap=3)`, `summarize_undercut_events` (adds `direction` undercut/overcut/ambiguous). Distance→time uses flat `REFERENCE_SPEED_MPS=83`. Four solved bugs (DriverAhead is a car *number*; use elapsed time not LapNumber across lapped cars; `"None"` string; check `is None` before `str()`).
* `rival-archive.py` (run all races), `enrich-archive.py` (+behind, direction, `points_gap`, `risk_appetite`), `archive-analysis.py`, `pit-loss.py` (diagnostic), `rival-evaluation.py` (validation candidate picker), `test.py` (Jolpica connectivity).
* `standings.py`: `get_signed_points_gap(year, race_fastf1, code_a, code_b)` / `get_points_gap` via Jolpica API (User-Agent required) with JSON cache.
* `risk.py`: `compute_risk_appetite(signed_gap)->"high"(|gap|≤25)|"medium"(≤75)|"low"|None` (heuristic, unvalidated).
* Validation summary: detector lands on real pit laps but **does not distinguish undercut intent** from traffic-forced/fastest-lap stops — treat as "pit-timing correlation". Per-race pit-loss constants range 3.4–51.7 s (outliers flagged; plausible band 8–35 s). Sprint session-name translation (`S`→`Sprint`) is an unverified guess; 25 known-backfilled sessions + 2021 Belgian GP excluded from analysis.

### 4.6 `driver/` (driver taxonomy = "Layer 2" profiles)
`overtake-defense.py` (POC event reconstruction from raw telemetry), `driver-taxanomy.py` (sampled-race aggression/defence, stratified by circuit type), `driver-taxanomy-metrics.py` (tyre_management, consistency_factor, wet_weather_skill via teammate-relative deltas), `pressure-tolerence.py` (clutch performance in last 3 races/season + self-inflicted DNF rate), `build-taxanomy.py` (merge → master), `driver-archetype-mapping.py` (<20 races → archetype: `<10` races rookie_aggressive/conservative split on self-inflicted-DNF rate, `≥10` junior_high_potential/senior_backmarker split on avg points/race; full-career static by design), `driver-rolling.py` + `join-profiles.py` (leakage-safe PTI versions for training), `tyre-regression-driver.py` (ablation: do PTI profiles improve Regression V2? results in `ablation_results.csv`/`ablation_paired_results.csv`), `calender.py` (2018–2026 `RACE_CALENDAR[(season,round)]` — author flagged "FACT-CHECK"), `race-order.py` (FastF1 schedule → global rank), `driver-metrics.py`/`telemetry-structure.py`/`test.py` (discovery/diagnostics).
Scales: metrics normalised to 0.85–1.15 (winsorised). **Nothing in the trees consumes driver profiles yet** — they are intended for Second-Driver logic and priority/risk.

### 4.6b NLP radio module (in `notebooks/`, not in `system/`)
`02-classify-radios.py` = keyword classifier (`CATEGORIES_DICT`, `HIGH_STRESS`, `MEDIUM_STRESS`; regex `\bphrase\b`; messages >500 chars → informational; stress only for `tyre_feedback`). `03-nlp-classifier.py` = validation: stress vs pit within 3 laps, control = informational tyre feedback → **no significant lift** (`stress_signal_lookahead_sweep.csv`, all p>0.37). Join key is `(Season, racing_number/DriverNumber)`; time alignment via `estimate_lap_start_times` (LapStartDate is null). **`get_driver_stress_trigger()` referenced in gate-tier-3.py does not exist as code.**

### 4.7 `strategic-sacrifice/` (team-order / "second driver" reward logic)
`reward.py`: `F1_POINTS`, `position_points`, `points_delta`, `DriverChampionshipState(name, wdc_gap, role_weight, title_secured)`, `TeamChampionshipState(wcc_gap, races_remaining, alpha=0.5)`, `championship_leverage(gap, races_remaining, title_secured)=1/(1+|gap|/(26·races_remaining))` (0 if secured), `compute_team_reward(...)`, `justification_check(R_sac, R_no_sac, tau=0.5)`. Validated on Brazil 2022 (VER/PER), Malaysia 2013 (Multi-21), Russia 2018 (BOT→HAM) with historical inputs; `sensitivity.py` shows flip points (Brazil: gap >~80 pts; Malaysia only flips if w1<w2). Role weights convention D1 0.65 / D2 0.35. **Not connected to `priority_driver_id`.**

### 4.8 Data pipeline (`system/cleaning`, `system/feature-engineering`, `system/eda`, `notebooks/`)
Download (fastf1 + tracinginsights + radios) → `laps-cleaning.py` (flags, `SC_CODE="4"`, `VSC_CODES={"6","7"}`, outlier factor 1.5× stint median) → `features.py` (engineered columns above; SC/VSC event lap detection from race-control messages: VSC ends with status `ENDING`, SC ends with `IN THIS LAP`) → `sanity-check.py`, `align-telemetry.py`, `circuit-taxanomy.py`, `weather-schema.py`, EDA scripts. `notebooks/fia-regulations-scaping.py` + `../fia-pdfs/*` (sporting regs 2018-2026) → `f1_tyre_constraints.json`.

---

## 5. WIRING MATRIX — every tree input, its source, and its status

Legend: **READY** = function exists and returns the right thing · **ADAPT** = exists but needs glue/derivation · **BUILD** = nothing exists.

### Tier 1
| Input | Source | Status | Notes |
|---|---|---|---|
| `red_flag_or_race_stopped` | `TrackStatus` contains `"5"` (replay) / live race-control | ADAPT | trivial |
| `tyre_structurally_damaged` | damage sensor / radio (`mechanical_issue`, "flat tyre", "damaged" in `HIGH_STRESS`) | BUILD | pick a rule, default False |
| `unsafe_weather` | weather modules | BUILD | define mapping: e.g. `CrossoverState.BOX_INTERS` (currently unreachable) or `DryingState.CONSIDER_DRIER_TYRE` while on wrong tyre, plus raining-now flag from FastF1 `Rainfall` |
| `tyre_age_history`, `laptime_seconds_history` | per-stint clean laps | ADAPT | keep only non-pit/non-SC/non-outlier laps of current stint |

### Tier 2
| Input | Source | Status |
|---|---|---|
| `season, circuit, is_sprint_weekend` | race config | READY |
| `compound_history_dry` | accumulate dry compounds seen this race (exclude INTER/WET) | ADAPT |
| `wet_race_exception` | driver ran INTER/WET (sensitivity-buffer.py's definition) or regs JSON | ADAPT |
| `laps_remaining_in_race` | `total_laps − lap` (total_laps: laps_features max LapNumber, or schedule) | ADAPT |
| `remaining_sets`, `sets_used_so_far`, `stops_made_so_far` | 2018/19 from FastF1 session data; 2020+ from allocation helpers | ADAPT (unused by `evaluate_tier2` today) |

### Tier 3
| Input | Source | Status |
|---|---|---|
| `cliff_probability_next_5_laps`, `predicted_pace_loss` | `build_tyre_life_projection` + `tyre_life_models.pkl` | READY after B1 fix; needs per-lap inputs: `fuel_load_estimate` (110·(1−(lap−1)/total)), `stint_number`, `track_temp_bucket`, `regulation_era=get_regulation_era(season)`, `circuit_degredation_ordinal` (taxonomy) |
| `tyre_age` | `TyreLife` | READY |
| `expected_stint_length` | **nothing** ("from strategy plan") | BUILD — options: Cox median survival for the stratum; median historical stint length per (compound, circuit, era) from `cliff_detection_stints.csv` (`n_laps_true`); Pirelli `lap_survival` 45/35/25; the team's own planned strategy |
| `safety_car_deployed` | `TrackStatus` contains `"4"` (or SC deployed message); decide VSC handling (`"6","7"`) | ADAPT |
| `undercut_opportunity` | `knowledge.flag_undercut_windows` logic | ADAPT/BUILD — the existing rule needs hindsight (§7-B4). Live rule: `gap_ahead_s < pit_loss_s` AND `tyre_age − ahead_tyre_age ≥ 3`, `pit_loss_s` from `archive_per_race_analysis.csv` per circuit |
| `overcut_opportunity` | none live | BUILD |
| `rival_undercut_threat` | mirror of undercut using the car **behind** (`behind_*` columns exist offline) | BUILD |
| `in_dirty_air` | `DistanceToDriverAhead / speed` < `CLOSE_FOLLOWING_SECONDS=1.0` (overtake-defense.py) | ADAPT |
| `driver_stress_signal` | classifier `02-classify-radios.py` + radio→lap mapping already in `03-nlp-classifier.py` (`estimate_lap_start_times`, `match_messages_to_nearest_lap`); "medium/high tyre_feedback within lookback window" | ADAPT — lift the mapping into `get_driver_stress_trigger(driver, lap, lookback)` (see B6) |
| `sc_gamble_recommendation` | `evaluate_sc_gamble(SCGambleInputs(...))` fed by `get_sc_probability` + projection | ADAPT (semantic fixes §7-B5) |

### Execution tree
| Input | Source | Status |
|---|---|---|
| `driver_id` "D1"/"D2" | map team's two 3-letter codes → D1/D2 (D1 = priority driver by convention) | ADAPT |
| `gate_tree_trigger_tier` | 1 if Tier-1 PIT_NOW, 2 if Tier-2 PIT_FLEXIBLE, 3 if Tier-3 PIT_NOW | ADAPT |
| `track_position` | `Position` | READY |
| `can_delay_one_lap_without_position_loss` | gap-behind vs pit-loss / rival analysis | BUILD (definition needed) |
| `tier3_reason`, `driver_stress_signal` | from `evaluate_tier3` output | READY |
| `safety_car_active` | `TrackStatus` | ADAPT |
| `priority_driver_id` | Second Driver logic | BUILD (candidates: `compute_team_reward` comparing "D1 priority" vs "D2 priority" with driver profiles; else None → default D1) |
| `circuit` | race config | READY |

### Loop closure
After each decision: increment `tyre_age`, append lap/compound history, update `stops_made`/`sets_used` if a pit was executed, refresh projection.

---

## 6. Input / output contracts to standardise in the orchestrator

**RaceContext (static per race):** `season, circuit (underscore name), total_laps, is_sprint_weekend, team, d1_code, d2_code, session, regulation_era, circuit_id, circuit_degredation_ordinal, pit_loss_s (empirical per circuit else 22.0 flagged), p_sc_5lap (circuit prior or None), rain_forecast_pct (optional)`.
**LapSnapshot (per driver per lap):** `lap, driver, compound, tyre_age, stint_number, position, gap_ahead_s, gap_behind_s, ahead_code/compound/tyre_age, behind_code/compound/tyre_age, lap_time_s, track_temp_c, track_status, is_pit_in/out flags, radio messages so far, rain_now, raining_forecast_pct, stops_made, sets_used, compound_history, stint_lap_history[(age, laptime)], remaining_sets`.
**DriverDecision (output):** `{lap, driver, tier_reached (1|2|3), gate_decision, reason, triggers{8}, projection{pace_loss, cliff_p}, sc_gamble{recommendation,cost_pit_now,cost_wait,saving}, execution{decision, penalty_seconds, driving_instruction, reason}, context{stress, weather_state, drying_state}, explanation_text, data_quality_notes[]}`.

---

## 7. KNOWN BUGS, INCONSISTENCIES & GAPS (fix these as part of the join)

**B1 — CONFIRMED: `track_temp_bucket` is silently ignored in the live tyre projection.** `tyre_life_projection.py` builds one-hot columns by comparing `col == f"track_temp_bucket_{bucket}"`, but the pickle's `temp_dummy_columns` are `temp_extreme/temp_hot/temp_warm` (built with `prefix="temp"`, `drop_first=True` so `cool` = all zeros). The comparison never matches ⇒ all temp dummies are always 0 ⇒ `cool/warm/hot/extreme` return **identical** output. Verified: MEDIUM Bahrain 2022-2025 age 15 gives 1.2465 s / p=0.01380 for all four buckets. 145 of 211 regression models have a non-zero temp coefficient (e.g. HARD Australian 2022-25 `temp_warm=+0.568`), and Cox has `temp_hot=+0.061, temp_extreme=−0.041`. **Fix:** match `col == f"temp_{bucket}"` in both `_regression_design_row` and `_cox_design_row` (and keep `regulation_era_` matching, which is correct). Add a regression test asserting four buckets give ≥2 distinct outputs for a model with non-zero temp coefficients.

**B2 — `get_driving_instruction` defined twice** in `execution-tree.py` (stub at line 55, real at line 61). Remove the stub.

**B3 — Hyphenated filenames** aren't importable (`gate-tier-1.py`, `gate-tier-2.py`, `gate-tier-3.py`, `execution-tree.py`, `sc-gamble.py`, `model-fit.py`, `sc-vsc-probability-model.py`, `rival-*.py`, `driver-*.py`, `enrich-archive.py`, …). Either loader shim (recommended for the one-file goal; keeps a single source of truth) or rename to snake_case. Comments elsewhere refer to the *old* underscore names (`gate_tree_tier3.py`, `sc_gamble_evaluator.py`, `cliff_detection.py`, `nlp_driver_stress.py`) — those files no longer exist under those names.

**B4 — The undercut detector uses hindsight.** `flag_undercut_windows` only evaluates laps within 3 laps *before that driver's real pit-in* (`in_pit_decision_window`). That is fine for archive analysis but cannot run live (you don't know the future pit lap). A live version must drop the window condition and apply `approx_gap_s < pit_loss_s AND tyre_age_gap ≥ 3` every lap; expect the ~37 % over-flagging the author saw before adding the window (need an extra guard, e.g. only when the driver's own tyre_age is ≥ some fraction of `expected_stint_length`, or gap < pit_loss − margin). `direction` (undercut/overcut) is *retrospective* pit-order sequencing and also can't be computed live. Distance→time uses a flat 83 m/s.

**B5 — SC-gamble input semantics are mismatched.**
* `predicted_pace_loss_per_lap` is documented as "per lap", but `predicted_pace_loss` from Regression V2 is the **absolute pace loss vs the stint's fastest clean lap at this tyre age** (e.g. 1.25 s), not a per-lap increment. Multiplying it by `n_laps_horizon` overstates waiting cost. Use the marginal slope `(loss(age+h) − loss(age))/h` or the average over the horizon minus the current value — call the projection at `tyre_age` and `tyre_age+h`.
* `cliff_probability_next_n_laps` must match the projection's `cliff_horizon_laps` (5) and `p_sc_next_n_laps` uses `HORIZON_LAPS=5`; the sc-gamble demos use `n_laps_horizon=3`. Use 5 everywhere or make the horizon a single config value.
* `p_sc` is a circuit-level constant (4–10 % per 5-lap window), so with `NORMAL_PIT_LOSS 22 / SC_PIT_LOSS 11` the gamble rarely says WAIT unless degradation is negligible — expected, but verify with a sweep.
* Constants 22.0 / 11.0 / 1.5 / 5.0 are assumptions. Real per-circuit pit loss exists in `archive_per_race_analysis.csv` (use it, and derive SC pit loss as a fraction).

**B6 — The stress pipeline is ~80 % built; only the per-lap wrapper `get_driver_stress_trigger()` (named in a gate-tier-3.py comment) is missing.** What already exists in `notebooks/`: (a) classification — `02-classify-radios.py` (`classify_radio_message`, `classify_stress_level`, keyword lists) → `classified_radios.csv`; (b) radio→lap mapping — `03-nlp-classifier.py`: `load_classified_radios` (tyre_feedback only, parses timestamps, derives season/race from `race_id`), `estimate_lap_start_times` (race start ≈ earliest radio timestamp of that race; lap start = start + cumulative prior `LapTime`, per driver, because `LapStartDate` is null except 2020) and `match_messages_to_nearest_lap` (joins on `(season, race, racing_number==DriverNumber)`, picks the lap whose estimated start is nearest the message time); (c) validation (no significant lift, so stress = context only). **What to build:** a function `stress_signal(driver, lap, lookback=N)` that lifts (b) out of the notebook, and returns True if a `medium`/`high` `tyre_feedback` message was matched to a lap in `[lap−N+1, lap]`. Caveats to handle: (1) "nearest lap start" can assign a message to the *next* lap (up to half a lap of look-ahead) — in replay use only messages matched to laps `<= current lap` to avoid leakage; (2) start-time drift after red-flag races; (3) only 2018+ races present in the CSV, only `tyre_feedback` has a stress level; (4) lookback N is undefined (sweep used 1/3/5/7).

**B7 — `expected_stint_length` has no producer** (see wiring matrix).

**B8 — `can_delay_one_lap_without_position_loss`, `overcut_opportunity`, `rival_undercut_threat` have no producer.**

**B9 — Regulation data disagree.** `f1_tyre_constraints.json` says `total_sets_allocated: 2` for 2020-2025 and `0` for 2026 and `mandatory_dry_compounds: 0` for 2026, while `gate-tier-2.py` uses 13/12 sets and **always requires 2 dry compounds** (era-independent — its own docstring says "constant 2018-2026"). For 2026 the JSON says no mandatory compound; if the JSON is right, Tier 2 will spuriously return `PIT_FLEXIBLE` near the end of 2026 races. Also `wet_race_exception` per JSON: 2018 true, 2019 false, 2020-2024 true, 2025 false — but `Tier2State.wet_race_exception` is a caller-supplied bool. Resolve against the FIA PDFs / MDP para 120-126 and make Tier 2 read one source.

**B10 — Tier 2 is thin:** `remaining_sets`, Q2 start-compound rule, Monaco 2025 two-stop, sprint-shootout set consumption exist as helpers but nothing enforces them (they should *filter the action space* / feed `Tier2State`); `PIT_FLEXIBLE` has no downstream meaning.

**B11 — Tyre-model coverage gaps.** 211 regression keys only; any missing `(compound, circuit, era)` → `pace_loss=None` → the trigger silently never fires. Add a fallback (era+compound pooled, then compound pooled) and a `data_quality_notes` entry. 2026+ strata are thin. Compound names are FastF1 strings; no INTER/WET model. Circuit keys must be the exact `Race` folder names (`70th_Anniversary_Grand_Prix`, `Sakhir_Grand_Prix`, `Styrian_Grand_Prix` appear as separate keys; aliases `Brazilian_/São_Paulo_`, `Mexican_/Mexico_City_`, `Dutch` etc. exist — see `CIRCUIT_ID_TO_RACE_NAMES` in `model-fit.py`; `archive-analysis.py` has `CIRCUIT_NAME_ALIASES`). The Cox `circuit_degredation_ordinal` requires the circuit→`circuit_id` map.

**B12 — Weather isn't wired and has no live wet-tyre model.** Tier 1 `unsafe_weather` needs a definition; `BOX_INTERS` is unreachable; Tier 3's tyre triggers are meaningless on INTER/WET (projection returns `None` ⇒ safe) but the decision then relies on weather modules alone.

**B13 — Duplicated logic drift risk** (`detect_cliff` ×3, `CLIFF_PROBABILITY_THRESHOLD` ×2, `CachedBucket` ×30). The orchestrator should import, then `assert` equality of duplicated constants at startup.

**B14 — `laps_features.gap_to_car_ahead` is a lap-boundary proxy** (session-time diff among cars on the same LapNumber), not a same-instant gap; the telemetry `DistanceToDriverAhead` is finer but only exists for the raw archive. `Position` was checked non-null only in 2023 Bahrain R (0 % null); the raw 2018 practice sample was 100 % null — verify per race before relying on it, else derive order from `gap_to_leader`.

**B15 — Live/replay clean-lap filter for Tier-1 cliff check** (see §4.1) — otherwise SC laps and pit laps false-trigger `tyre_cliff_already_hit`.

**B16 — Project hygiene:** `f1-race-engineer/` untracked in git; `calender.py` self-flagged as unverified; Sprint session translation unverified; 25 backfilled sessions.

---

## 8. THE ORCHESTRATOR — build spec for `system/HERMES/hermes_master.py`

**Recommendation (decide with the user):** one file that *loads* the existing modules through an `importlib` shim (single source of truth) and contains adapters + the per-lap engine + a CLI. Fall back to inlining only if the user insists on zero external files. Either way keep the author's "None-safe, flag assumptions, explain decisions" style.

### 8.1 File layout
```
hermes_master.py
  0. CONFIG / constants (paths, horizon=5, thresholds re-exported + asserts)
  1. load_module(path, name)                      # hyphen-safe loader
  2. Static loaders (cached): tyre pickle, circuit taxonomy → {race→(circuit_id, degr_ordinal, type, double_stack_risk…)},
                              SC prior csv, per-circuit pit loss csv, driver taxonomy, radios csv, regs JSONs
  3. Adapters (pure functions, each returns (value, note)):
       derive_features(...), project_tyres(...), tier1_inputs, tier2_inputs,
       live_undercut/overcut/rival_threat/dirty_air, stress_trigger, sc_gamble,
       weather_state, expected_stint_length, can_delay_one_lap, second_driver_priority
  4. Engine: run_lap(ctx, snapshots) -> list[DriverDecision]
  5. Sources: ReplaySource (laps_features + opponents checkpoints + radios + weather.csv) ; LiveSource stub (same interface)
  6. Reporting: console table + JSONL/CSV per lap + explanation strings
  7. CLI: --year --race --session R --team "Red Bull Racing" --d1 VER --d2 PER --laps 1-57 --explain --out
  8. Self-checks (--selftest): B1 regression test, constant-sync asserts, golden scenarios
```
### 8.2 Per-lap algorithm (pseudo-code)
```python
for lap in laps:
    for d in (D1, D2):
        s = snapshot(d, lap)
        feats = derive_features(ctx, s)          # era, fuel, temp bucket, degr ordinal
        proj  = build_tyre_life_projection(..., **feats)   # after B1 fix; fallback if None
        # ---- Tier 1
        t1 = evaluate_tier1(Tier1State(red_flag(s), damaged(s), unsafe_weather(ctx, s),
                                       clean_stint_ages(s), clean_stint_laptimes(s)))
        if t1 == "PIT_NOW": result[d] = gate("PIT_NOW", tier=1); continue
        # ---- Tier 2
        t2 = evaluate_tier2(Tier2State(...))
        if t2 == "PIT_FLEXIBLE": result[d] = gate("PIT_FLEXIBLE", tier=2); continue
        # ---- Tier 3
        gamble = evaluate_sc_gamble(SCGambleInputs(p_sc, marginal_pace_loss, proj.cliff, 5))  # only needed if a tyre trigger is live
        t3 = evaluate_tier3(Tier3State(proj.cliff, proj.pace_loss, s.tyre_age, expected_stint_length(...),
                                       undercut, overcut, sc_active, rival_threat, dirty_air,
                                       driver_stress_signal=stress, sc_gamble_recommendation=gamble.recommendation))
        if t3.decision == "PIT_NOW": tier=3 -> triggered
        else: driving_instruction = get_driving_instruction(t3.decision, proj.cliff, t3.reason, stress)
    triggered = [DriverPitContext(...) for d in (D1,D2) if gate PIT_NOW/PIT_FLEXIBLE]
    if triggered: exec_out = evaluate_execution_tree(ExecutionTreeState(triggered, sc_active, circuit, priority_id))
    merge exec_out + non-triggered drivers' instructions -> DriverDecision rows
    update histories (tyre_age+1, compound_history, stops, sets)
```
Rules: never let an exception in one adapter kill the lap (catch → record `data_quality_notes`, treat the input as `None`/False as the trees intend). Log `tier_reached`, all 8 triggers, and a plain-English `explanation`. Keep D1/D2 mapping explicit; when the tree says both trigger, the tie-break order is Second-Driver override → lower tier → D1.

### 8.3 Adapters — exact definitions the builder must choose (and flag as ASSUMPTION)
| Adapter | Proposed definition |
|---|---|
| `unsafe_weather` | raining-now (FastF1 `Rainfall`) while on slicks, OR `CrossoverState.CONSIDER_INTERS/BOX_INTERS` while `rain_now`; else False |
| `expected_stint_length` | median `n_laps_true` from `cliff_detection_stints.csv` for (compound, circuit, era), fallback compound+era median, fallback Pirelli `lap_survival`; record which fallback |
| `live_undercut` | `gap_ahead_s < pit_loss_s` and `tyre_age − ahead_tyre_age ≥ 3` (no hindsight window); `gap_ahead_s = distance_to_ahead_m/83` or `gap_to_car_ahead` |
| `rival_undercut_threat` | car behind within `pit_loss_s` and its tyre_age ≥ 3 laps *fresher*… mirror of above |
| `overcut_opportunity` | rival ahead has pitted/likely to pit and our tyres still healthy (cliff_p below threshold) and clean-air gap — define & flag |
| `in_dirty_air` | `gap_ahead_s < 1.0` (CLOSE_FOLLOWING_SECONDS) |
| `stress_trigger` | any `tyre_feedback` with `stress_level ∈ {medium, high}` in the last `N` laps (N=3 default; sweep used 1/3/5/7) |
| `can_delay_one_lap` | `gap_behind_s > pit_loss_s + margin` **or** no rival within undercut range behind; flag as ASSUMPTION |
| `priority_driver_id` | v1: `None` (→ D1 default). v2: compare `compute_team_reward` for both orderings using `driver_taxonomy_master_final.csv` + standings (`standings.get_signed_points_gap`) and `risk.compute_risk_appetite` |
| `marginal_pace_loss` | `(proj(age+5).pace_loss − proj(age).pace_loss)/5`, floor 0 |

### 8.4 Testing plan (golden cases)
1. **B1 regression test:** 4 temp buckets ⇒ non-identical outputs for `('HARD','Australian_Grand_Prix','2022-2025')` at age 15.
2. Re-run every tree file's `__main__` scenarios through the shim (all documented outputs must be unchanged).
3. Constant-sync asserts: `gate-tier-3.CLIFF_PROBABILITY_THRESHOLD == execution-tree.CLIFF_PROBABILITY_THRESHOLD == 0.017`; `get_regulation_era` vs pickle eras.
4. Replay **2023 Bahrain (RBR: VER/PER)** end to end; sanity-check the pit laps the tree recommends against actual `is_pit_in` laps (RBR real stops) — report agreement ± laps, don't claim accuracy.
5. Replay **2022 Monaco** (wet→dry; real switch laps: Gasly ~4, Hamilton/Perez 16, Norris/Leclerc/Verstappen 18) to exercise weather adapters.
6. A red-flag race (2021 Saudi / 2023 Australia) → Tier 1 fires, compound history preserved.
7. Sprint weekend + 2018 (missing allocation → `None` path) + 2026 (regs mismatch B9).
8. Missing-model scenario (`WET`) → no crash, notes recorded.
9. Double-stack scenarios (SC vs no SC) → penalties 5.0/0.0 as in `execution-tree.py` demos.

### 8.5 Runtime facts
* Loading `tyre_life_models.pkl` needs the same `lifelines`/`scikit-learn` versions it was pickled with (venv). Cox `predict_survival_function` per call is slow-ish; cache per (compound,age,…) if replaying many laps.
* Replay needs no GCS if `gcs_cache/` is present (all `clean/features` cached, 894 `weather.csv` cached). Raw telemetry JSON only exists locally for 2018 Abu Dhabi practice; for opponent gaps use `checkpoints/rival_knowledge/*_opponents(_enriched).csv` (173–178 races).
* Windows paths: use `pathlib`; repo cwd is `C:\Users\larai\f1-race-engineer\f1-race-engineer`.

---

## 9. Constants ledger — validated vs assumed

| Constant | Value | Where | Status |
|---|---|---|---|
| `CLIFF_PROBABILITY_THRESHOLD` | 0.017 | tier3, execution | **Calibrated** (ROC on 533 real cliffs, AUC 0.683, Youden point TPR 0.65/FPR 0.395; deliberately permissive since 3+ triggers are needed). Was 0.30. |
| `PACE_LOSS_THRESHOLD_SECONDS` | 1.805 | tier3 | **Calibrated** = P90 of pace_loss over 148,041 clean dry laps. |
| `TYRE_AGE_TRIGGER_RATIO` | 0.421 | tier3 | **Calibrated** = P10 of cliff_tyre_age/n_laps_true over 611 cliffs (after the n_laps undercount fix; max ratio now 0.927). |
| `DEADLINE_BUFFER_LAPS` | 5 | tier2 | Evidence-backed via compliance-margin sweep (flags 0.41 % of compliant driver-races). |
| Tier-3 vote rule | ≥3 PIT_NOW, 1–2 PIT_LATER | tier3 | Design choice (from diagram), not fitted. |
| `MIN_SEGMENT_LENGTH/MIN_IMPROVEMENT/MIN_SLOPE_RATIO/MIN_STEP_SECONDS` | 3 / 0.20 / 2.0 / 0.3 | cliff detect | Tuned against real false positives. |
| `INSTRUCTION_NEUTRAL_BAND_RATIO` | 0.5 | execution | **ASSUMPTION** |
| `DOUBLE_STACK_PENALTY_SECONDS` | 5.0 | execution | From MDP para 130 |
| SC-position rule | worse-placed car pits first under SC | execution | **ASSUMPTION** (flagged) |
| `NORMAL_PIT_LOSS / SC_PIT_LOSS / RESTART_COLD_TYRE / cliff_penalty` | 22 / 11 / 1.5 / 5×p s | sc-gamble | **ASSUMPTIONS** |
| SC circuit prior | `p_window_horizon` per circuit | csv | **Validated as circuit-level only** (lap-level AUC 0.503) |
| `HORIZON_LAPS` | 5 | SC prior, cliff feed | convention |
| Rain crossover `(monitor,consider)` early/mid/late | (45,60)/(30,45)/(20,35) | weather | **Provisional** |
| Drying gate / evidence | 300 s / −1.0 s | weather | **Provisional** |
| Risk bands | 25 / 75 pts | risk.py | **Heuristic**, unvalidated |
| `CLOSE_FOLLOWING_SECONDS` | 1.0 | driver | DRS-based assumption |
| Undercut `window_laps`, `min_tyre_age_gap`, `REFERENCE_SPEED_MPS` | 3, 3, 83 | knowledge.py | Starting guesses |
| Fuel model | 110 kg linear | features.py | Proxy assumption |
| Reward `α, w1/w2, τ, λ` | e.g. 0.3–0.6, 0.65/0.35, 0.1, 1.0 | strategic-sacrifice | Case-specific, set per scenario |
| `MIN_ROWS_PER_GROUP` | 40 | model-fit | matches regression v2 |
| Cox penalizer / min events | 0.1 / 15 | model-fit | design |

## 10. Validation evidence you can quote
Cliff ROC AUC 0.683 (n=533). Cox concordance 0.619. KM: compound survival differs (log-rank p<0.0001). SC lap-level prior AUC 0.503 [0.475,0.533] (rejected); circuit-level incidence sanity-checked (Baku/Jeddah 100 %, Monaco 62 %, Singapore 67 %). NLP stress vs pit: no significant lift at 1/3/5/7-lap lookaheads (all p>0.37). Rival detector: lands on real pit laps in 2023 Bahrain/2019 Monaco/2018 Abu Dhabi checks but not intent-specific. Reward function: 3 historical cases (Brazil 2022, Malaysia 2013, Russia 2018) behave as expected; sensitivity flip points documented. Weather: Sochi 2021 + Monaco 2022 backtests; synthetic monotonicity/stage-order checks. Driver ablation: see `ablation_results.csv` (Regression V2 baseline vs +TM/+Consistency/+Both, paired group bootstrap).

## 11. Open questions to put to the user (answer before/while building)
1. One self-contained file (inline everything) or one entry file that loads existing modules? (Recommend the latter.)
2. Replay-only first (historical races) or live-style streaming input too?
3. Which team/drivers are D1/D2 for default runs (Red Bull VER/PER for 2022–2024?), and how should D1 be chosen per race?
4. 2026 rules: trust the JSON (0 mandatory compounds) or the code (always 2)?
5. Definitions for `can_delay_one_lap`, `overcut_opportunity`, `rival_undercut_threat`, `unsafe_weather`, `tyre_structurally_damaged` (§8.3 proposals).
6. `expected_stint_length` source (historical median vs Cox vs team plan).
7. Should `PIT_FLEXIBLE` behave differently from `PIT_NOW` in Execution?
8. Should the SC-gamble use per-circuit empirical pit loss instead of 22/11?
9. Is Second-Driver priority in scope for v1 (default D1) or should the reward function drive it?

## 12. Commands (from repo dir `C:\Users\larai\f1-race-engineer\f1-race-engineer`)
```
.\.venv\Scripts\python.exe system\HERMES\trees\execution-tree.py          # self-tests
.\.venv\Scripts\python.exe system\HERMES\trees\gate-tier-3.py
.\.venv\Scripts\python.exe system\HERMES\trees\model-fit.py --output tyre_life_models.pkl   # needs GCS creds + cliff_detection_stints.csv
.\.venv\Scripts\python.exe system\HERMES\trees\test.py                    # run from the folder containing tyre_life_models.pkl
```

## 13. Glossary of state strings
Tier 1 → `PIT_NOW | MOVE_TO_TIER_2`. Tier 2 → `PIT_FLEXIBLE | MOVE_TO_TIER_3`. Tier 3 → `PIT_NOW | PIT_LATER | DONT_PIT` (+reason `SC_GAMBLE | SOFT_TRIGGERS | None`). SC gamble → `WAIT | NO_ADVANTAGE_TO_WAITING | INSUFFICIENT_DATA`. Execution decision → `PIT_NOW | PIT_LATER`; double-stack modes `DOUBLE_STACK_BY_POSITION | AVOID_DOUBLE_STACKING_STAGGER_1_LAP`; driving instruction → `PIT_LAP | MANAGE | NEUTRAL | PUSH | CONSERVE`. Weather → `STAY_SLICKS | MONITOR | CONSIDER_INTERS | BOX_INTERS(unreachable)`; drying → `STAY_WET | GATE_OPEN_NO_EVIDENCE | CONSIDER_DRIER_TYRE`. Risk → `high | medium | low`. Undercut direction → `undercut | overcut | ambiguous`. Era → `2018-2021 | 2022-2025 | 2026+`. Temp bucket → `cool | warm | hot | extreme`.
