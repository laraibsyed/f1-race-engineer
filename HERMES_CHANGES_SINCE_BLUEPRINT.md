# HERMES — Changes Since the 2026-09-26 Blueprint/Code Bundle

> Companion to `HERMES_MASTER_BLUEPRINT.md` / `HERMES_CODE_BUNDLE.md` (the 2026-09-26 snapshot,
> §0 TL;DR of which ended with: *"Build ONE runnable file that joins every finished module to the
> Gate Tree and Execution Tree."*). That file was built. This document is the "what happened since"
> info dump: everything added, fixed, validated, and built on top of it, in the order it happened,
> with exact file paths and the reasoning behind each change. Nothing here is guessed — every
> number quoted was produced by actually running the code.
>
> Generated 2026-10-01, manually (not scripted) from the actual session history.

---

## 0. TL;DR

The blueprint's open questions (§11) are now answered by code, not just discussion:

1. **One entry file that loads existing modules** — `system/HERMES/trees/master.py` (~2,034 lines), using exactly the hyphen-safe `importlib` loader the blueprint recommended (§8.1/§B3).
2. **Replay-only, historical** — `master.py replay` CLI, no live-streaming input built.
3. **D1/D2 default** — resolved per-season from `evaluate.py`'s `RBR_PAIRS` table (now covers 2018–2026, not just 2021–2025 — see §9).
4. **B1 (track_temp_bucket bug) — FIXED.** Confirmed via a regression test: 4 temp buckets now give ≥2 distinct outputs.
5. **B2 (duplicate `get_driving_instruction`) — FIXED** as part of the master.py build.
6. **`can_delay_one_lap`, `overcut_opportunity`, `rival_undercut_threat`, `unsafe_weather`, `tyre_structurally_damaged`** — all have live definitions now (§4).
7. **`expected_stint_length`** — sourced from `cliff_detection_stints.csv` historical medians (§4).
8. Two whole new subsystems the blueprint didn't anticipate: a **Monte Carlo SC-gamble evaluator** (§2) and **genuine tabular Q-learning RL layer** (§3), both strictly additive (flag-gated, never replacing the validated analytic/Gate-Tree path).
9. A **multi-stage evaluation program** (§5–§7) that went from "does HERMES roughly agree with history" to "is that agreement statistically distinguishable from chance" to "does the system's own documented limitations hold up."
10. A **full Streamlit demo dashboard** ("HERMES Pit Wall", §8) — not in the original blueprint at all, built as a pure orchestration/UI layer with zero changes to decision logic.

---

## 1. `master.py` — the orchestrator itself (the blueprint's actual deliverable)

**New file:** `system/HERMES/trees/master.py`.

Built exactly along the lines of blueprint §8.1:

- `load_module(path, name)` — the hyphen-safe loader (§8.1 step 1), used to load `gate-tier-1.py`,
  `gate-tier-2.py`, `gate-tier-3.py`, `execution-tree.py`, `tyre_life_projection.py`,
  `sc-gamble.py`, and (optionally, `required=False`) `weather/crossover.py`, `weather/drying_line.py`,
  `rival-awareness/risk.py` — all loaded once at import time as module-level attributes
  (`gate1`, `gate2`, `gate3`, `exec_tree`, `tyre_proj`, `sc_gamble`, `crossover`, `drying_line`, `risk_mod`).
- `RaceContext` / `DriverRuntimeState` dataclasses (blueprint §6 "RaceContext"/"LapSnapshot" contracts,
  simplified to what the engine actually needed) — `RaceContext` holds season/circuit/total_laps/
  d1_code/d2_code/session/regulation_era (auto-derived via `get_regulation_era`)/
  circuit_degredation_ordinal/pit_loss_s/p_sc_5lap; `DriverRuntimeState` carries the lap-to-lap
  accumulating state (compound history, stint ages/laptimes, stops made) the blueprint's §6
  LapSnapshot described as needing somewhere to live.
- `evaluate_driver_lap(ctx, state, row, lap_df, resources, radios_by_driver_lap, data_quality_notes, laps=None)`
  — the per-driver, per-lap engine: walks Tier 1 → Tier 2 → Tier 3 exactly as the blueprint's §8.2
  pseudocode specified, building the live adapters it flagged as BUILD/ADAPT (see §4 below).
- `merge_execution(ctx, results, safety_car_active)` — calls `execution-tree.py`'s
  `evaluate_execution_tree` for whichever driver(s) triggered PIT_NOW/PIT_FLEXIBLE this lap.
- `run_replay(ctx, resources, lap_range, ...)` — the blueprint's §4.5 "ReplaySource", reading
  `laps_features.csv` directly (no GCS calls needed once `gcs_cache/` is populated, as §8.5 predicted).
- **Explainability layer** (blueprint §6 "explanation_text" output) — `explanation_for(r)`,
  `explanation_text_for(r)`, `plain_english_for(r)`, `top_contributing_signals(r, n=3)`,
  `gate_path_for(r)`. `top_contributing_signals` does real threshold-margin ranking (how far past
  its own threshold each active Tier-3 trigger is), not dict order.
- **`selftest()`** (blueprint §8.4) — now 22 checks (see §9/§10 for what was added to it over time),
  all passing. Run via `master.py selftest`.
- CLI: `master.py replay --season --race --d1 --d2 --session --laps --out --explain
  [--sc-gamble analytic|mc] [--rl] [--rl-qtable]`.

**Resolved, not just flagged:**
- B9 (regulation JSON vs code disagreement) — the code's "always 2 dry compounds, era-independent"
  reading was kept; not reconciled against `f1_tyre_constraints.json`, but the discrepancy is no
  longer silent — `circuit_degredation_ordinal_for`/`pit_loss_for_circuit` log which source matched.
- B13 (duplicated-constant drift risk) — `selftest()` includes an explicit
  `CLIFF_PROBABILITY_THRESHOLD in sync (tier3 vs execution-tree)` check.
- B14 (`Position` reliability) — `_rank_by_gap_to_leader(lap_df)` fallback added: when `Position`
  isn't usable, ranks by `gap_to_leader` instead, used everywhere adjacency matters.

---

## 2. Monte Carlo Safety-Car Gamble (additive, frozen)

**Modified file:** `system/HERMES/safety-car/sc-gamble.py` (backed up first as
`sc-gamble_pre_*` per the project's backup convention before any edit).

Per explicit instruction to keep this *strictly scoped* (don't redesign the analytic evaluator,
wire behind a flag, A/B test, don't over-optimize):

- Added `evaluate_sc_gamble_mc(inputs, config=None, pit_durations=None)` — Monte Carlo uncertainty
  propagation around the *same* cost model the original `evaluate_sc_gamble` already used, sampling
  pit-stop durations from circuit-centred, per-circuit-group 10/90-winsorized empirical pools
  (`pit_stop_durations.csv`), not a synthetic distribution.
- `master.py` gained `--sc-gamble {analytic,mc}` (env override `HERMES_SC_GAMBLE`); **default stayed
  `analytic`** — nothing about production behaviour changed unless the flag is passed.
- A/B harness: `system/HERMES/safety-car/sc-gamble-ab.py` — runs both evaluators across the same
  grid of scenarios, checks same return schema, same WAIT/NO_ADVANTAGE/INSUFFICIENT_DATA semantics,
  reproducible with a fixed seed, decisions broadly agree, no change to the rest of HERMES.
- **Explicitly documented limitation, not hidden:** the "2.59s simulated spread matches 2.59s
  empirical spread" result was documented as a consistency check on the winsorization
  *implementation*, not independent validation (both numbers come from the same convention).
- **Frozen** after this — no further tuning, per explicit instruction once the comparison passed.

---

## 3. Reinforcement Learning layer (additive, advisory-only)

**New files:** `system/HERMES/trees/rl_env.py`, `rl_agent.py`, `rl_train.py`, `rl_bridge.py`,
`rl_validate.py`, `rl_artifacts/qtable.json`.

Built to a detailed spec requiring **genuine tabular Q-learning** (not a heuristic dressed up as
RL), mandatory backups before touching any existing file, and honest reporting of negative results.

- `rl_env.py` — `HermesStrategyEnv`, a simulated environment that reuses HERMES's *real* fitted
  tyre models (not a toy reward function) via a projection cache keyed on
  (compound, circuit, era, stint, temp, degr, tyre_age, fuel) for speed.
- `rl_agent.py` — `TabularQAgent` with the real Bellman update
  `Q(s,a) += α(r + γ·maxQ(s',·) − Q(s,a))`, discretized HERMES-derived state space.
- `rl_bridge.py` — `rl_recommend_for_hermes_result(agent, result, ctx)` +
  `combine_with_gate_tree(rl_out, result)` — **advisory only**: the RL recommendation is attached
  as `result["rl"]`, alongside the real Gate/Execution Tree decision, never overriding it. Wired
  into `master.py replay --rl` as a lazy, opt-in import (`master.py` has zero import-time dependency
  on the RL layer when the flag isn't passed).
- `rl_validate.py` — compares 4 policies over 200 matched-seed episodes (disjoint eval seed base
  9000 vs training seed 42): random, untrained (fresh Q-table), **`always_stay_out`** (explicit
  degenerate baseline), and trained. 16 pass/fail regression checks, including a structural
  no-lookahead check and confirmation `master.py selftest` still passes after the RL layer exists.

### Reward redesign (the honest negative-result fix)

First training pass: the trained policy did **not** beat the degenerate `always_stay_out` baseline.
Root cause, confirmed by direct computation: a single Bahrain MEDIUM pit stop (~23.685s) costs more
than an entire stint's cumulative degradation (~18–20s over 15–20 laps) — so "never pit" was
reward-optimal absent a mandatory-compliance constraint. Per instruction to fix the **reward
formulation**, not tune until results look good:

- `STAY_OUT_LOOKAHEAD_LAPS=3` — forward-looking mean pace-loss cost added to the reward.
- `MANDATORY_COMPOUND_PENALTY_SECONDS = sc_gamble.NORMAL_PIT_LOSS_SECONDS` — terminal penalty via
  `gate2.mandatory_compound_done(...)`, reusing the *existing* regulatory constant, not an invented
  bonus.
- **Final, reported-as-is result:** 3,000 episodes, mean return −427.57s → −121.43s, training
  compliance 100% → 87%, but only **38% on held-out eval seeds (9000–9199)** — reported honestly as
  a generalization gap, not smoothed over or hidden.

---

## 4. Live adapters the blueprint flagged as BUILD/ADAPT — now implemented in `master.py`

Direct resolution of the blueprint's §4–§5 wiring matrix gaps:

| Blueprint gap | Resolution in `master.py` |
|---|---|
| `unsafe_weather` (B12) | `unsafe_weather(rain_now, on_slicks, rain_probability_pct, laps_remaining, total_laps)` — reads real `Rainfall` (attached via `_attach_weather_columns`, merge_asof, no lookahead) |
| `expected_stint_length` (B7) | Historical median `n_laps_true` from `cliff_detection_stints.csv`, per (compound, circuit, era), via `expected_stint_length(...)` |
| `live_undercut`/`rival_undercut_threat` (B4) | `live_undercut_opportunity`/`rival_undercut_threat` — **no hindsight**: requires the neighbour to have a *real, already-happened* `is_pit_in` within a recency window (`UNDERCUT_RECENCY_WINDOW_LAPS = HORIZON_LAPS`), **not** the archive's future-pit-lap window the blueprint flagged as non-live-safe |
| `in_dirty_air` | wired from adjacency (`ahead`/`behind` gap) via `_rank_by_gap_to_leader` |
| `driver_stress_signal` (B6) | `build_radios_by_driver_lap(repo_root, laps, ctx)` lifts the notebook mapping (`estimate_lap_start_times`/`match_messages_to_nearest_lap`) into a per-lap live lookup; gracefully degrades to `False` all race if `classified_radios.csv` has no rows for that race |
| `sc_gamble_recommendation` | `run_sc_gamble(inputs)` wraps whichever evaluator is active (analytic/mc) |
| `marginal_pace_loss` (§8.3) | `marginal_pace_loss(build_projection, tyre_age, horizon=HORIZON_LAPS)` |
| B1 fix verification | `selftest()` check: *"B1 fix: 4 temp buckets give >=2 distinct pace_loss outputs"* |

---

## 5. Undercut false-positive fix + causal-attribution follow-up

Two rounds, each backed up first (`master_pre_undercut_calibration_fix_backup.py`).

**Round 1 — recency fix.** Root cause, found by raw-data tracing: `live_undercut_opportunity`/
`rival_undercut_threat` originally only checked `gap < pit_loss_s AND age_gap >= 3` — a **static**
starting-tyre-age difference (not a real pit event) could satisfy this for an entire race. Added
`driver_pitted_within_window(laps, driver_code, current_lap, window_laps=UNDERCUT_RECENCY_WINDOW_LAPS)`
— requires a genuine `is_pit_in` within the window, backward-only (no lookahead).

**Round 2 — causal attribution.** The recency-only fix left one residual false positive (PER L13,
Bahrain 2023) where a neighbour's real pit *coincided with* an already-existing 3-lap gap but didn't
*create* it. Fixed with `neighbor_advantage_created_by_recent_pit(...)`: compares `gap_before`
(at the neighbour's pit-in lap, using the pre-reset TyreLife value) vs `gap_after` (now), requiring
`gap_before < UNDERCUT_MIN_TYRE_AGE_GAP <= gap_after` — the pit must have carried the gap *across*
the threshold, not just coincide with an already-large one.

Both rounds added to `selftest()`: 8 undercut/calibration checks + 5 "Cases A–E" causal tests
(all passing), including explicit no-lookahead regression tests (a future pit must be invisible
until `current_lap` reaches it).

---

## 6. Bahrain circuit-degradation / pit-loss calibration fix

Found during a deep, read-only diagnostic into persistent early `PIT_NOW` for VER/PER at Bahrain
2023 (confirmed real, not caused by the undercut bug above — both issues coexisted).

- `circuit_degredation_ordinal_for(taxonomy, race_folder_name)` — now tries `circuit_id` match
  first (via a `CIRCUIT_ID_TO_RACE_NAMES` alias table duplicated from `model-fit.py`, same
  "duplicate deliberately" convention the blueprint documents), falling back to substring match.
  Bahrain now resolves to ordinal 2 (high), matching taxonomy row `'Sakhir International Circuit'`,
  not the medium(1) fallback.
- `pit_loss_for_circuit(pit_loss_table, race_folder_name, season=None)` — normalizes
  underscore/space naming, and (new) accepts an optional `season` to disambiguate the archive's
  one-row-per-year-per-circuit structure (previously always picked the first/earliest year's value
  via `.iloc[0]`). Bahrain 2023 now resolves to the real ~25.35s archive value instead of the
  22.0s `DEFAULT_PIT_LOSS_SECONDS` fallback.
- **Quantitatively confirmed non-decisive for the original Bahrain symptom** (both bugs were real,
  but the undercut false-positive was the dominant cause of the persistent early PIT_NOW).

---

## 7. Multi-stage evaluation program

Four stages, each explicitly building on — not replacing — the last.

### Stage A — `evaluate.py` (the main evaluation framework; pre-existing-but-extended)
`system/HERMES/trees/evaluate.py` — reused throughout everything below:
- `RBR_PAIRS`/`rbr_drivers_for` (see §9 — fixed/extended in this session)
- Walk-forward folds (`fold1_2021`…`fold4_2024`) + genuine `HOLDOUT_YEAR = 2025`, both leaky
  (`eval_out/`) and clean/refit-per-fold (`eval_out_clean/`, via `model-fit-fold.py` +
  `audit.py`'s future-leakage checks) variants
- `baseline_2of3` — the designated fair baseline: a tyre-life-only ablation of HERMES's own
  Tier-3 triggers (2-of-3 of cliff_proximity/pace_lap_delta/tyre_age), no rival/SC/radio/weather
- `counterfactual_time_delta` — alternate-pit-lap estimate (equal pit loss assumed, first-order)
- Null controls (`stage_controls`): uniform-random timing, same-circuit-other-season
- Paired bootstrap (`paired_full_vs_baseline.csv`): full system beats baseline,
  mean ΔF1 = +0.147 (clean), 95% CI [0.102, 0.191], excludes 0

### Stage B — `strategic_window_validation.py` (Stage 1 + Stage 2)
**New file**, deliberately *not* duplicating `evaluate.py`'s F1/precision-recall machinery — answers
a different, more literal question ("did HERMES flag the window becoming strategically inferior
within ±3 laps of the real pit").
- Stage 1: 10 objectively-selected races, Window Recall 100%, Pre-Pit Recall 97.3%, Exact
  Agreement 18.9%.
- **Critical finding:** base rate of PIT_NOW/PIT_LATER across *all* decision laps is 80.7% —
  meaning the ±3 window metric is close to tautological on this dataset, not strong evidence of
  timing accuracy.
- Stage 2 extension (same file, same backup-before-edit convention —
  `strategic_window_validation_pre_stage2_backup.py`): recommendation-onset/warning-lead
  distribution (median 16 laps before the pit), escalation metrics, last-pre-pit-state
  distribution, trigger-count-delta, and a **saturation-aware baseline** (same prevalence, no
  trigger info) that reproduces the *same* 100% window recall — confirming the Stage 1 metric's
  saturation quantitatively, not just qualitatively.

### Stage C — `final_evaluation_summary.py` (closure)
**New file** — explicitly a *consolidator*, not a new experiment: reads already-computed CSVs from
`eval_out_clean/` and `eval_out_strategic_window/`, computes nothing new, and assembles the single
final evidence table + 5-question interpretation the dissertation closure required (paired
performance, fold stability, 2025 holdout, counterfactual, timing error, null-control
distinguishability). Per the task's own stop condition: no further validation frameworks proposed
after this.

---

## 8. HERMES Pit Wall — the Streamlit demo dashboard

Not in the original blueprint at all — an entirely new orchestration/UI layer built on top of
everything above, with the hard constraint (honoured throughout, verified by `master.py selftest`
after every round) that **no HERMES decision-logic file is ever modified** to make the UI easier.

**New directory:** `dashboard/` — `hermes_adapter.py`, `data_loader.py`, `track_map.py`, `app.py`,
`tests/test_dashboard.py`, `requirements.txt`.

### Core adapter (`hermes_adapter.py`)
Imports `master.py`/`evaluate.py` exactly as they exist and calls only their already-public
functions (`evaluate_driver_lap`, `merge_execution`, `explanation_for`, `resolve_red_bull_pair` →
wraps `evaluate.rbr_drivers_for`, never re-derives it). `ReplayCache`/`ScenarioReplayCache` give
O(1) backward lap navigation and forward-only incremental computation, with scenario branches
(SC/VSC/weather injection) forked from a frozen historical-state snapshot so they never mutate
`bundle.laps` (verified by dedicated tests).

### Genuine-telemetry track map (`track_map.py`)
Went through two full implementations:
1. **v1:** FastF1 live/cached telemetry per lap, one static snapshot per lap (functional, but
   produced one "jump" per lap rather than motion).
2. **v2 (current):** a proper client-side HTML5 Canvas renderer — real per-driver recorded
   (time, X, Y) tracks (`np_tracks`) sent to the browser once, animated via
   `requestAnimationFrame`, interpolating only *between actually-recorded telemetry samples*
   (never fabricated coordinates). Zero Python-side flicker during playback, since the browser
   animates independently of Streamlit's own rerun cycle. If genuine telemetry can't be obtained
   for a race/session, the UI shows `TRACK GEOMETRY UNAVAILABLE` rather than any invented shape.

### Bug-fix history (each confirmed against the live dashboard, each re-verified against
`master.py selftest` = 22/22 and the full `dashboard/tests/test_dashboard.py` suite)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Sidebar showed nothing after a prior collapse | CSS `header[data-testid="stHeader"]{display:none}` also hid the sidebar's own re-expand toggle, which lives in that header | Keep the header element present, just slim/transparent instead of `display:none` |
| 2 | SC/VSC scenario "never went away" | Two real bugs: (a) `ScenarioReplayCache` crashed (`KeyError: 0`) when injected at lap 1, because lap 0 has no decision row, only a state snapshot; (b) the top-bar SC/VSC indicator and "SCENARIO ACTIVE" badge ignored the scenario's own duration, staying on forever once triggered | (a) fork from `historical._state_after[fork_from]` directly, never `historical.dpc` (which keeps mutating); (b) added `sc_window_active` bounded by `scenario_lap + duration - 1`, used consistently for both the badge and the SC/VSC readout |
| 3 | Driver-card text (WHY / triggers / SC GAMBLE) rendered overlapping each other | Many separate `st.markdown()` calls relying on Streamlit's own inter-element gap, which wasn't applying in this Streamlit version | Rewrote each driver card as a single HTML block with explicit `margin-top` on every line — no longer dependent on Streamlit's own spacing |
| 4 | Strategy Timeline chart effectively invisible | 70px height, no axis labels | 120px height, driver-code y-axis labels, lap-number x-axis title |
| 5 | A driver's headline decision (e.g. `PIT_LATER`) looked inconsistent with its own "WHY" text (e.g. "PIT_NOW because...") | Not a bug — `master.explanation_for()`'s `plain_text` always explains the **Gate Tree's** raw decision, while the headline shows the **Execution Tree's** final call after driver-priority tie-break; both numbers were correct, just unexplained side by side | Added a note, shown only when they diverge: *"Gate-tree call was X; Execution Tree adjusted to Y (driver-priority / execution rules)"* |
| 6 | Track map (esp. portrait-shaped circuits like Las Vegas) rendered pushed to one side | `scaleanchor` keeps 1:1 aspect but doesn't centre the shrunk axis by default | `constrain="domain"` + `constraintoward="center"/"middle"` on both axes |
| 7 | D1/D2 raised `KeyError` for 2018/2019/2020/2026 | `RBR_PAIRS` only covered 2021–2025 | See §9 |

### What stayed deliberately out of scope
Never touched: Gate Tree, Execution Tree, tyre/cliff model, undercut logic, SC gamble, weather
model, RL — the dashboard is purely an orchestration/adapter layer calling the exact same
functions `master.py`'s own CLI calls.

### Launch
```
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

---

## 9. Red Bull driver-pairing table — extended (this session's most recent fix)

**Modified file:** `system/HERMES/trees/evaluate.py` (backed up first as
`evaluate_pre_rbr_pairs_backup.py`, hash-verified byte-identical).

`RBR_PAIRS` previously covered only 2021–2025 — any other season raised a `KeyError`
(surfaced in the dashboard as *"No Red Bull pairing recorded for 2026"*). Every entry below was
verified against the real `Team` column in that season's own `laps_features.csv` (every single
race checked individually for 2018/2020/2026; every 2019 race checked individually for the
mid-season driver swap) — nothing assumed from memory:

| Season | Pairing | Verified how |
|---|---|---|
| 2018 | VER + RIC | all 20 races checked |
| 2019 | VER + GAS (rounds 1–12) → VER + ALB (round 13, Belgian GP, onward) | all 21 races checked individually; the exact 12/9 split is a new `RBR_2019_ALB_RACES` exception set, same pattern as the existing 2025 `RBR_2025_EARLY_LAW_RACES` |
| 2020 | VER + ALB | all 17 races checked |
| 2021–2024 | VER + PER | unchanged |
| 2025 | VER + LAW (rounds 1–2) → VER + TSU | unchanged |
| 2026 | VER + HAD | all 6 available races checked |

All 12 season/exception combinations re-verified against `rbr_drivers_for` after the edit;
full regression suite re-run (14/14 dashboard tests, 22/22 `master.py selftest`).

---

## 10. Constants ledger — net changes since the blueprint's §9 table

| Constant | Blueprint status | Now |
|---|---|---|
| `track_temp_bucket` matching (B1) | Confirmed broken (always matches 0) | **Fixed**, regression-tested in `selftest()` |
| `get_driving_instruction` duplicate def (B2) | Present | **Removed** |
| Undercut `window_laps`/causal check | Hindsight-only (B4), flagged non-live-safe | **Live, causal**: recency window + gap-before/gap-after attribution |
| `CLIFF_PROBABILITY_THRESHOLD` / `PACE_LOSS_THRESHOLD_SECONDS` / `TYRE_AGE_TRIGGER_RATIO` | 0.017 / 1.805 / 0.421 (calibrated) | **Unchanged** — never retuned by any of the work above |
| `DEADLINE_BUFFER_LAPS` | 5 | **Unchanged** |
| SC gamble cost constants (22.0/11.0/1.5) | Assumption | **Unchanged** (analytic path); MC path adds empirical pit-duration sampling around the same model, doesn't replace these |
| `RBR_PAIRS` | 2021–2025 only | **2018–2026**, see §9 |
| Tier-3 vote rule (0 / 1–2 / 3+) | Design choice | **Unchanged** |

Nothing in this ledger was retuned "until results looked good" at any point — every change above
either fixed a confirmed bug (traced to a specific root cause, backed by before/after numbers) or
added a genuinely new, validated data source (Monte Carlo pit-duration pools, RL's real tyre-model
reuse, the circuit-id alias table, the per-season RBR verification).

---

## 11. File inventory — everything new or modified since the blueprint, in one place

**New:**
- `system/HERMES/trees/master.py` (+ `master-old.py`, `master_pre_rl_backup.py`,
  `master_pre_undercut_calibration_fix_backup.py` — backups)
- `system/HERMES/trees/rl_env.py`, `rl_agent.py`, `rl_train.py`, `rl_bridge.py`, `rl_validate.py`
  (+ `*_pre_reward_redesign_backup.py` for each, `rl_artifacts/qtable.json`)
- `system/HERMES/safety-car/sc-gamble-ab.py`
- `system/HERMES/trees/strategic_window_validation.py`
  (+ `strategic_window_validation_pre_stage2_backup.py`)
- `system/HERMES/trees/final_evaluation_summary.py`
- `system/HERMES/trees/audit.py`, `model-fit-fold.py`, `experiment.py`, `cliffs.py` (evaluation
  infrastructure referenced throughout §7)
- `dashboard/` (entire directory — `hermes_adapter.py`, `data_loader.py`, `track_map.py`, `app.py`,
  `tests/test_dashboard.py`, `requirements.txt`)
- `.claude/launch.json` (dashboard dev-server config)
- This file, `HERMES_CHANGES_SINCE_BLUEPRINT.md`

**Modified (each with a byte-identical, hash-verified backup made first):**
- `system/HERMES/safety-car/sc-gamble.py` — added `evaluate_sc_gamble_mc`, kept
  `evaluate_sc_gamble` and its constants unchanged
- `system/HERMES/trees/evaluate.py` — `RBR_PAIRS` extended (§9); no other changes to its
  evaluation methodology

**Never modified:** `gate-tier-1.py`, `gate-tier-2.py`, `gate-tier-3.py`, `execution-tree.py`,
`tyre_life_projection.py`, `model-fit.py`, `threshold-calibration.py`, `sensitivity-buffer.py`,
`sc-vsc-probability-model.py`, `weather/crossover.py`, `weather/drying_line.py`,
`rival-awareness/knowledge.py`, `rival-awareness/risk.py`, `rival-awareness/standings.py`,
`strategic-sacrifice/reward.py`, any tyre-fitting script, any taxonomy file. Every threshold in
§9's ledger that was already calibrated in the blueprint stayed exactly as calibrated.
