# HERMES — Team Strategy Layer, Dashboard Integration & Repo Cleanup

> Companion to `docs/HERMES_CHANGES_SINCE_BLUEPRINT.md` (which covers everything up to and
> including the original Streamlit "HERMES Pit Wall" dashboard and the RL layer). This document
> picks up from there: the full team-coupled strategic layer (12 phases), the critical fix that
> made it operationally wired into the live Gate/Execution Tree rather than merely advisory, the
> dashboard integration that followed (risk mode as a real control, decision trade-off
> explainability, override visibility), a performance fix uncovered while wiring the dashboard,
> and a final repo-wide cleanup pass.
>
> Generated 2026-10-01/02, manually from the actual session history. Every number quoted was
> produced by actually running the code.

---

## 0. TL;DR

1. **New subsystem: the Team Strategy Layer** — `system/HERMES/trees/team-strategy.py` (814
   lines), a pure-function module implementing all 12 phases of a team-coupled strategic layer
   (role assignment, rival selection/prediction, joint D1×D2 candidate simulation, Monte Carlo,
   team reward, risk modes, regulatory feasibility, quantified explanation) on top of HERMES's
   *existing* models — no new tyre/weather/SC model, no duplicate Gate/Execution Tree logic.
2. **It is operationally wired, not just advisory.** A selected joint strategy can now
   genuinely change a driver's `gate_decision` (upgrade a soft Tier-3 call to `PIT_NOW`), while
   never touching Tier 1/2 and never silently suppressing an already-active Tier-3 `PIT_NOW` —
   any such conflict is reported as an explicit `override`, never applied silently.
   `system/HERMES/trees/master.py`'s `apply_team_strategy_to_lap()` is the exact, single place
   this happens; the real, unmodified Execution Tree runs immediately afterward.
3. **Rival intelligence is a real scoring input**, not just an explanation field — a predicted
   rival pit window now shifts a candidate's time-equivalent cost (`rival_pit_window_adjustment`),
   so covering vs. being exposed to an undercut genuinely changes which candidate ranks first.
4. **Risk mode is now exposed on the dashboard as a real control** — CONSERVATIVE / BALANCED /
   AGGRESSIVE propagate from a sidebar toggle all the way into candidate ranking (verified: the
   same lap under CONSERVATIVE vs AGGRESSIVE produces different `risk_adjusted_score`s).
5. **New explainability layer**: "WHY THIS DECISION?" trade-off cards (WHAT / WHY / alternative /
   net advantage / cost-vs-gain comparison table), built strictly from the already-ranked
   candidates — no second simulator, no fabricated numbers.
6. **Execution overrides are now visible on the dashboard**, not just in the JSONL — a
   TEAM STRATEGY vs EXECUTION panel shows when the Execution Tree accepted vs. overrode the
   recommendation, and why.
7. **Performance fix**: per-lap tyre-projection calls inside the team-strategy candidate loop
   were being recomputed ~25×/lap for identical inputs (an existing, pre-accepted cost in the
   CLI path that only became painful once the dashboard started calling it every lap) — a
   `functools.lru_cache` wrapper cut a 10-lap dashboard replay from 14.3s to 0.6s (~24×) with
   zero change to any computed number.
8. **Full repo reorganisation** — 12 superseded "backup"/"-old" files pulled out of the live
   `system/HERMES/trees/` folder into `trees/backups/`; 30 EDA charts, 26 derived CSVs, 10 logs,
   and 3 blueprint docs moved out of the repo root into `eda_charts/`, `analysis_outputs/`,
   `logs/`, and `docs/` respectively. Nothing the live pipeline actually loads (5 files, verified
   by grep) was moved.

---

## 1. The Team Strategy Layer — 12 phases, one new file

**New file:** `system/HERMES/trees/team-strategy.py` (814 lines).

Zero file I/O, zero network calls, zero imports of any other HERMES module — the same
standalone-callable convention `gate-tier-3.py` / `execution-tree.py` already use. `master.py`
owns gathering real inputs (standings, projections, `expected_stint_length`, driver taxonomy,
pit loss, adjacency) and calling these pure functions once per lap.

| Phase | Function(s) | What it does |
|---|---|---|
| 1 | `assign_team_roles` | Picks `priority_driver_id` / `support_driver_id` / `team_objective`. Resolution order: compromised car → championship leverage gap (`ROLE_LEVERAGE_FLIP_MARGIN = 0.15`) → genuine on-track position → default D1-priority fallback (byte-identical to the pre-existing `resolve_second_driver_priority_v1` default when nothing fires). |
| 2 / 6 | `generate_candidate_actions`, `joint_candidates` | Small, fixed per-driver action menu (`PIT_NOW` / `PIT_IN_3` / `PIT_IN_6` / `EXTEND_TO_CLIFF` [+`ALT_COMPOUND` when the mandatory-compound rule isn't satisfied yet]), evaluated as the **D1 × D2 Cartesian product** — genuinely joint, not two independent single-car decisions. |
| 3 | `evaluate_team_reward` | Thin wrapper around `strategic-sacrifice/reward.py`'s `compute_team_reward` — reused verbatim, not reimplemented. |
| 4 | `identify_relevant_rivals` | Distinguishes "nearest car" from "strategically relevant rival" from real on-track adjacency + the real championship-standings rival — never hardcodes a team/driver identity. |
| 5 | `predict_rival_pit_window` | Heuristic (explicitly **not** a calibrated model): ratio of the rival's own tyre age to its own `expected_stint_length`, boosted by the rival's own cliff probability. Returns a `[window_start, window_end]` + `confidence`, or `None` if inputs are missing — never a guess. |
| 5 (wiring) | `rival_pit_window_adjustment` | Makes the predicted window a genuine **scoring** input: ±fraction of the real, circuit-specific `pit_loss_s` added to a candidate's `time_delta_s` depending on whether its pit lap covers, lands inside, or is exposed past the rival's window (`COVER_BONUS=0.10`, `INSIDE_PENALTY=0.05`, `EXPOSED_PENALTY=0.20`, all fractions of real pit loss, not invented constants). |
| 7 | `apply_driver_profile` | Folds `driver_taxonomy_master_final.csv`'s `tyre_management` into the projected degradation (direct divisor on the model's own marginal pace-loss output); `aggression_level`/`pressure_risk_tolerance` used elsewhere (track-position trade-off, risk-mode penalty respectively). |
| 8 | `track_position_pace_tradeoff` | Upgrades the boolean dirty-air trigger into a quantitative cost, reusing the same `CLOSE_FOLLOWING_SECONDS` threshold the live `in_dirty_air` trigger already uses. |
| 9 | `apply_risk_mode` | Re-scores each candidate by subtracting a variance penalty (`RISK_MODE_VARIANCE_PENALTY = {CONSERVATIVE: 2.0, BALANCED: 1.0, AGGRESSIVE: 0.4}`) scaled by the Monte Carlo coefficient-of-variation — **genuinely changes the ranking**, not a cosmetic label. |
| 10 | `filter_regulatory_feasible` | Drops joint candidates that would make the mandatory 2-dry-compound rule impossible to satisfy — reuses `gate2.mandatory_compound_done` verbatim. Never returns an empty candidate set. |
| 11 | `monte_carlo_outcome` | Perturbs the already-computed deterministic point estimate (Gaussian noise on degradation, Bernoulli cliff event) — does **not** re-run the tyre model N times. `n_sims=200` by default (vs. `sc-gamble.py`'s 20,000 — deliberately cheap since this runs per candidate per lap). |
| 12 | `build_strategic_explanation` | Only reports numbers actually present on the selected/alternative candidates — never fabricates a percentage or figure the simulation didn't produce. |
| orchestrator | `evaluate_team_strategy` | The one entry point `master.py` calls per lap: role → candidates → regulatory filter → simulate (+ rival adjustment) → Monte Carlo → team reward → risk-mode ranking → explanation, in that order. |

**Driver taxonomy integration status** (`driver_taxonomy_master_final.csv`, 0.85–1.15 neutral-1.0
scale): `tyre_management` → degradation scaling, `aggression_level` → dirty-air relief,
`pressure_risk_tolerance` → risk-mode penalty are wired in. `defensive_strength` and
`wet_weather_skill` are **not** used — flagged in the code as a documented gap, not silently
dropped.

---

## 2. The critical fix: making the selected strategy *operational*

### 2.1 The gap

The first version of the Team Strategy Layer (described in §1 above) computed a genuinely joint,
rival-aware, risk-adjusted strategy — but it stopped at `r["team_strategy"] = team_strategy_result`.
The real decision (`gate_decision`, `execution.driving_instruction`) was computed independently
by the existing Gate/Execution Tree and never looked at it. Calculated, but not connected.

### 2.2 The fix

**New function**, `system/HERMES/trees/team-strategy.py::resolve_strategy_execution` (pure,
~75 lines with docstring) — the missing link between "we calculated a strategy" and "it
operationally happened." Asymmetric, safety-respecting rule:

- **Tier 1 (hard safety) or Tier 2 (regulatory)** already decided this driver's fate this lap →
  team strategy **never** overrides it. A conflicting recommendation is reported as an explicit
  `override`, never applied.
- Team strategy's action is `PIT_NOW` and the Gate Tree's own Tier-3 result was `DONT_PIT` or
  `PIT_LATER` (no hard constraint, no already-active trigger) → **UPGRADE**: `executed_action`
  becomes `PIT_NOW`. Not a bypass — it works by flipping `gate_tree_trigger_tier` to 3 so the
  driver becomes `triggered`, and the **real, unmodified Execution Tree** then runs for them
  exactly as it would for any other `PIT_NOW` (double-stack/priority logic included).
- Team strategy wants to stay out, but Tier 3 **already** independently triggered `PIT_NOW` from
  real signals (cliff risk, pace loss, tyre age, undercut) → **not suppressed**. Reported as an
  explicit override — a speculative team-level simulation does not get to talk a driver out of a
  trigger-justified pit stop.
- Otherwise → no change, no override, reported for transparency.

**Wiring**, `system/HERMES/trees/master.py::apply_team_strategy_to_lap` (extracted 2026-10-03 so
both the CLI replay loop and the dashboard adapter share the exact same code — see §4):
called between `evaluate_driver_lap` and `merge_execution`, mutates `gate_decision` /
`gate_tree_trigger_tier` only on an upgrade, and always attaches
`r["team_strategy_execution"] = {team_strategy_action, executed_action, override, override_reason,
should_upgrade_to_pit_now}` for both drivers.

### 2.3 Live proof (Bahrain 2023, VER/PER)

```
L12 VER: upgrade  → gate_decision=PIT_NOW → execution.driving_instruction=PIT_LAP
L13 VER: upgrade  → gate_decision=PIT_NOW → execution.driving_instruction=MANAGE   (Execution Tree's
                                             own double-stack logic staggered it — correct, not a bug)
L5  PER: override → team wanted stay-out, Tier-3 PIT_NOW preserved, reason reported
```

### 2.4 Tests

8 new tests added to `system/HERMES/trees/test_team_strategy.py`: upgrade, non-suppression of an
active Tier-3 trigger, Tier-1/2 untouchable, no-op when aligned, rival-window-changes-ranking, and
one true end-to-end test (`test_end_to_end_selected_strategy_reaches_real_execution_tree`) that
replays real laps and asserts against genuine Gate Tree output, not mocks.

---

## 3. Dashboard integration — risk mode, trade-off explainability, override visibility

### 3.1 Audit finding

Before this pass, `dashboard/hermes_adapter.py::evaluate_lap` had its **own**, separate per-lap
evaluation loop (`evaluate_driver_lap` + `merge_execution` directly) that never called the team
strategy layer at all — zero risk-mode control, zero team-strategy output, on the dashboard,
despite it being fully wired into the CLI replay path (§2).

### 3.2 Risk mode as a real control

`dashboard/hermes_adapter.py`:
- `DriverPairContext` gained `team_strategy_fn` / `team_strategy_unavailable_reason` fields,
  built automatically by `build_driver_pair_context` (gracefully `None` if the module or
  championship-standings fetch is unavailable — never crashes the page).
- `ReplayCache` / `ScenarioReplayCache` gained `set_risk_mode(mode)` — clears only the **decision**
  cache (driver-state evolution, i.e. tyre age/stops, is risk-mode-independent — proven by a
  dedicated test), so a mode change never forces a full re-walk from lap 1.
- `master.py::build_team_strategy_context`'s inner closure gained an additive
  `risk_mode_override` kwarg — lets a single already-built context be re-evaluated under a
  different mode per call, without re-fetching championship standings or rebuilding taxonomy.

`dashboard/app.py`: a `st.radio` RISK MODE control in the sidebar, wired straight to
`replay_cache.set_risk_mode(...)`.

**Verified propagation** (same lap, same candidates): CONSERVATIVE gave `risk_adjusted_score =
-7.872`, AGGRESSIVE gave `-7.8144` on an identical input — genuinely different, confirming the
control is wired, not cosmetic. (On that particular window the top-ranked candidate didn't flip —
an explicitly accepted, explicitly tested outcome, not a bug.)

### 3.3 Decision trade-off explainability

**New function**, `team-strategy.py::build_decision_trade_off_explanation` — built strictly from
the already-ranked `selected` candidate and its strongest `alternatives[0]` (the same objects
`evaluate_team_strategy` already produced — no second simulator). Reports:

- **WHAT**: selected D1/D2 action, pit lap, horizon (`pit_lap - current_lap`, the model's own
  real evaluated horizon — never a fixed invented number of laps).
- **WHY**: degradation cost, cliff penalty, pit cost, dirty-air cost, rival-window adjustment —
  all read directly off the real `d1_outcome`/`d2_outcome` dicts.
- **Alternative** and **net advantage** (`alternative.time_delta_s - selected.time_delta_s`, both
  already in the same unit — seconds — from the same cost model, never a fabricated cross-unit
  comparison).
- A **comparison table** (immediate pit cost / degradation / track position / rival effect /
  risk-adjusted score, selected vs. alternative).
- **Execution status** — the `resolve_strategy_execution` override info passed straight through,
  so overrides are never silently hidden from the explanation.

Wired into `apply_team_strategy_to_lap` as `team_strategy_result["trade_off_explanation"]`, so
both the CLI JSONL output and the dashboard get it from the same call.

`dashboard/ui_components.py`: `risk_mode_context_html`, `team_role_html`, `trade_off_card_html`
render this as a "WHY THIS DECISION?" expander per driver, plus a TEAM ROLE panel
(`D1 — CHAMPIONSHIP / TEAM PRIORITY` tag) and a TEAM STRATEGY vs EXECUTION status block
(FOLLOWED / OVERRIDE + reason) added to the existing strategy card.

### 3.4 Tests

6 new tests in `test_team_strategy.py` (trade-off fields, no-fabrication spot-checks, rival
effect surfacing, execution-status pass-through, risk-mode-override-changes-result). 7 new tests
in `dashboard/tests/test_dashboard.py` (team-strategy attached by default, execution consistency,
risk-mode propagation, cache-invalidation-is-decisions-only-not-state, trade-off presence,
graceful degradation when team-strategy is unavailable).

---

## 4. Shared wiring: `apply_team_strategy_to_lap`

Previously the strategy→execution wiring (§2.2) was inlined once, inside `master.py::run_replay`'s
per-lap loop. Factored out (2026-10-03) into `master.py::apply_team_strategy_to_lap(...)` —
a behaviour-preserving refactor (verified: full regression suite unchanged before/after) — so the
CLI replay path and the dashboard adapter call the **exact same code**, not two copies of the same
logic. `run_replay` now calls it; `dashboard/hermes_adapter.py::evaluate_lap` calls it too.

---

## 5. Performance fix: redundant tyre-projection calls

While wiring the dashboard, a 10-lap `ReplayCache.get()` walk with team-strategy enabled profiled
at **14.3 seconds**. `cProfile` traced it to `tyre_life_projection.build_tyre_life_projection`
(a `lifelines` Cox-model `predict_cumulative_hazard` call) being invoked **1,560 times across 10
laps** — 156 calls/lap, because every one of the ~16–25 joint candidates evaluated per lap called
`pace_loss_fn(d.tyre_age)` fresh, even though `d.tyre_age` is the **same value** for every
candidate that lap (it only changes once per lap, not per candidate).

**Fix**, `master.py::build_team_strategy_context._driver_sim_inputs`: wraps the per-lap projection
closure in `functools.lru_cache(maxsize=None)`, recreated fresh every lap (no cross-lap staleness
risk). Pure memoisation of a deterministic function — changes **zero** computed numbers, confirmed
by the full test suite passing unchanged before and after.

**Result**: 10-lap dashboard replay 14.3s → 0.6s (~24×); `test_team_strategy.py` 24s → 4.2s;
full `dashboard/tests/test_dashboard.py` suite ~183s → 23s.

*(Also fixed along the way: a championship-standings network fetch with no bound could hang the
dashboard's "LOAD RACE" for the real retry/backoff budget of `rival-awareness/standings.py`
[~2 minutes] if the network was unavailable — wrapped in a `TEAM_STRATEGY_CHAMP_CTX_TIMEOUT_S = 8.0`
daemon-thread timeout, specific to the dashboard's interactive use; `standings.py`'s own retry
policy is untouched for the CLI batch path.)*

---

## 6. Repo-wide cleanup

### 6.1 `system/HERMES/trees/backups/` (new folder)

12 superseded files moved out of the live `trees/` folder (confirmed via grep: nothing imports
them by path, only stale comments referenced a couple of them):

```
evaluate_pre_rbr_pairs_backup.py          master_pre_rl_backup.py
master-old.py                             master_pre_team_strategy_backup.py
master_pre_dashboard_integration_backup.py master_pre_undercut_calibration_fix_backup.py
rl_env_pre_reward_redesign_backup.py      rl_validate_pre_reward_redesign_backup.py
rl_train_pre_reward_redesign_backup.py    strategic_window_validation_pre_stage2_backup.py
team-strategy_pre_dashboard_integration_backup.py
test_team_strategy_pre_dashboard_integration_backup.py
```

### 6.2 Repo root — before/after

**Root-level clutter moved** (76 git-tracked files total):

| New folder | Contents |
|---|---|
| `eda_charts/` | All 30 root-level `.png` files (lap-time, degradation, SC/VSC, undercut, correlation, cliff-detection, survival, sensitivity plots). |
| `analysis_outputs/driver_profiles/` | 7 CSVs: taxonomy roster/bucket, rolling profiles, pressure-risk-tolerance, simple metrics, joined laps+profiles. |
| `analysis_outputs/validation/` | 12 CSVs: ablation, SC-probability (full/LOO), deadline-buffer, stint-length/z-threshold sensitivity, tyre-regression v1/v2, SC/VSC empirical prior. |
| `analysis_outputs/data_quality/` | 8 files: feature sanity, telemetry coverage/flattened profile, data dictionary, inventory, GCS audit, clean-laps summary, candidate-races debug. |
| `analysis_outputs/race_specific/` | 6 files: British GP pit decisions + weather-wired JSONL, 4 overtake/defense event logs. |
| `logs/` | 10 files: RL train/validate probe + reward-fix logs, strategic-window run logs, loose decision JSONL dumps. |
| `docs/` | `HERMES_CHANGES_SINCE_BLUEPRINT.md`, `HERMES_CODE_BUNDLE.md`, `HERMES_MASTER_BLUEPRINT.md` (this file now lives here too). |

**Left at repo root, deliberately** — verified via `grep` that each is loaded through a
`REPO_ROOT`-anchored or `_data_path(REPO_ROOT, ...)` path by `master.py`, `dashboard/*.py`, or the
test suite; moving any of these would have broken the live pipeline:

```
tyre_life_models.pkl              driver_taxonomy_master_final.csv
cliff_detection_stints.csv        pit_stop_durations.csv
sc_vsc_circuit_level_prior.csv
```

Plus the standard `.env`, `.gitignore`, `README.md`, `service-account.json`.

### 6.3 Verification after the move

Full regression suite re-run after every reorganisation step: `test_team_strategy.py` 40/40,
`master.py selftest` 22/22, `dashboard/tests/test_dashboard.py` 20/20, Brazil 2022 → SACRIFICE /
Malaysia 2013 → HOLD POSITIONS both still correct. Nothing was committed to git as part of the
move — working-tree only, pending review.

---

## 7. File inventory (this document's scope only)

| File | Status | Lines |
|---|---|---|
| `system/HERMES/trees/team-strategy.py` | new | 814 |
| `system/HERMES/trees/master.py` | modified | 2,430 |
| `system/HERMES/trees/test_team_strategy.py` | new | 702 |
| `dashboard/hermes_adapter.py` | modified | 551 |
| `dashboard/app.py` | modified | 579 |
| `dashboard/ui_components.py` | modified | 309 |
| `dashboard/tests/test_dashboard.py` | modified | 313 |

Backed up before every edit this session (hash-verified): `master_pre_team_strategy_backup.py`,
`master_pre_dashboard_integration_backup.py`, `team-strategy_pre_dashboard_integration_backup.py`,
`test_team_strategy_pre_dashboard_integration_backup.py` (all now under `trees/backups/`, §6.1).

**Not touched**: Gate Tree, Execution Tree, `reward.py`, `rival-awareness/`, tyre/weather/SC
models, `strategic-sacrifice/validation.py` — confirmed by diff against pre-session backups and
by the unchanged Brazil 2022 / Malaysia 2013 validation output.

---

## 8. Known limitations (carried forward honestly, not smoothed over)

- Rival pit-window prediction and its scoring adjustment are explicitly heuristic — not a
  calibrated classifier. Documented as such in the code and in every explanation string.
- Championship-standings fetch degrades to "unavailable" (default-priority fallback) after an
  8-second dashboard timeout if the Jolpica API is unreachable; the CLI path still uses
  `standings.py`'s own full retry budget.
- `net_advantage_s` / the trade-off comparison table are D1-anchored (the car rival intelligence
  is built against); D2's own cost breakdown is shown but not separately compared to an alternative.
- `defensive_strength` and `wet_weather_skill` from the driver taxonomy remain unused in the
  Team Strategy Layer — flagged, not silently dropped.
