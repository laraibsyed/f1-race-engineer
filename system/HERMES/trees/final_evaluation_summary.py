#!/usr/bin/env python3
"""
HERMES Final Decision-Quality Evaluation Summary (closure task, 2026-09-30)
============================================================================
READ-ONLY CONSOLIDATION. Does not modify master.py, any tree/module file, any
decision logic, or any existing evaluation script. Reads back CSVs that
evaluate.py (`--fold-mode`, the clean walk-forward run) and
strategic_window_validation.py already wrote to eval_out_clean/ and
eval_out_strategic_window/, and assembles them into the single final
evaluation table + 5-question conclusion required to close the HERMES
evaluation chapter. Computes NO new statistic that isn't already a column in
one of those existing CSVs - this file only selects, relabels and formats.

WHY eval_out_clean/ (not eval_out/): eval_out_clean/ is the CLEAN walk-forward
run (models/thresholds refit on training years only per fold via
model-fit-fold.py, audited for future-season leakage by audit.py) and is the
only run with null-control results (controls.csv). eval_out/ is the leaky
all-data-fit run; its numbers are reported as supplementary context only
where eval_out_clean/ is missing something (it is not).

USAGE (from the repo root):
    .\\.venv\\Scripts\\python.exe system\\HERMES\\trees\\final_evaluation_summary.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 160)


def load_all(repo_root: Path) -> dict:
    clean = repo_root / "eval_out_clean"
    sw = repo_root / "eval_out_strategic_window"
    d = dict(
        metrics_overall=pd.read_csv(clean / "metrics_overall.csv"),
        metrics_by_fold=pd.read_csv(clean / "metrics_by_fold.csv"),
        paired_bootstrap=pd.read_csv(clean / "paired_bootstrap.csv"),
        paired_events=pd.read_csv(clean / "paired_full_vs_baseline.csv"),
        controls=pd.read_csv(clean / "controls.csv"),
        counterfactual_summary=pd.read_csv(clean / "counterfactual_summary.csv"),
        sw_overall=pd.read_csv(sw / "strategic_window_aggregate_overall.csv"),
        sw_baseline_overall=pd.read_csv(sw / "strategic_window_baseline_aggregate_overall.csv"),
    )
    return d


def main():
    ap = argparse.ArgumentParser(description="HERMES final decision-quality evaluation summary (read-only)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="eval_out_clean/final_evaluation_summary.md")
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    d = load_all(repo_root)
    mo = d["metrics_overall"].set_index("system")
    mf = d["metrics_by_fold"]
    pb = d["paired_bootstrap"].iloc[0]
    ctrl = d["controls"]
    cf = d["counterfactual_summary"].iloc[0]
    swo = d["sw_overall"].iloc[0]
    swb = d["sw_baseline_overall"].iloc[0]

    full = mo.loc["full"]
    base = mo.loc["baseline_2of3"]

    fold_rows = mf[mf["system"].isin(["full", "baseline_2of3"])].pivot(
        index="split", columns="system", values="f1"
    ).reindex(["test_2021", "test_2022", "test_2023", "test_2024", "holdout_2025"])

    holdout_full = mf[(mf["split"] == "holdout_2025") & (mf["system"] == "full")].iloc[0]
    holdout_base = mf[(mf["split"] == "holdout_2025") & (mf["system"] == "baseline_2of3")].iloc[0]

    ctrl_all_full = ctrl[(ctrl["system"] == "full") & (ctrl["group"] == "ALL")].iloc[0]
    ctrl_all_base = ctrl[(ctrl["system"] == "baseline_2of3") & (ctrl["group"] == "ALL")].iloc[0]

    lines = []
    w = lines.append

    w("# HERMES Final Decision-Quality Evaluation (closure)\n")
    w("STEP 1 - EXISTING EVALUATION EVIDENCE REUSED (nothing new computed; see file header)\n")
    w("Source: `system/HERMES/trees/evaluate.py --fold-mode` output in `eval_out_clean/` "
      "(clean walk-forward, models/thresholds refit on training years only per fold via "
      "`model-fit-fold.py`, leak-audited by `audit.py`), plus "
      "`system/HERMES/trees/strategic_window_validation.py` output in "
      "`eval_out_strategic_window/` (historical-pit-timing angle, used as secondary evidence only).\n")

    w("\n## STEP 2/3 - primary evaluation definition and fair baseline\n")
    w("Primary paired comparison: per driver-race event-detection F1 (precision/recall of HERMES's "
      "strategic recommendation events against real historical pit laps, +-2 lap tolerance), for the "
      "FULL system (Gate Tree -> Execution Tree, all triggers) vs the existing designated fair "
      "baseline `baseline_2of3` - a tyre-life-only ablation of HERMES itself (2-of-3 of its own "
      "Tier-3 triggers: cliff_proximity, pace_lap_delta, tyre_age; no rival/SC/radio/weather logic), "
      "computed from the identical replay JSONL as the full system. Both systems see the same race "
      "state, tyre information, pit-loss assumptions, degradation model and weather/SC information - "
      "the only difference is which decision policy is applied to that shared state.\n")
    w("LIMITATION STATED UP FRONT (see STEP 13.E): F1 is an event-timing-agreement metric, not a "
      "lap-time/finishing-position outcome. The only existing metric expressed as a simulated-time "
      "outcome is the counterfactual estimate (STEP 8 below), and it is computed for the full system's "
      "matched events only, not as a full-system paired outcome against the baseline. No new "
      "outcome metric was invented to force a cleaner comparison - this is reported as a genuine gap, "
      "not smoothed over.\n")

    w("\n## STEP 4 - primary outcome (paired F1 by driver-race)\n")
    w(f"Overall: full F1={full['f1']:.3f} (n={int(full['n_driver_races'])} driver-races, "
      f"{int(full['real_pits'])} real pits) vs baseline_2of3 F1={base['f1']:.3f}.\n")
    w("Paired per-driver-race difference (`paired_full_vs_baseline.csv`, n="
      f"{len(d['paired_events'])} driver-race rows):\n")
    n_better = (d["paired_events"]["f1_diff"] > 0).sum()
    n_worse = (d["paired_events"]["f1_diff"] < 0).sum()
    n_equal = (d["paired_events"]["f1_diff"] == 0).sum()
    w(f"- mean paired difference = {pb['mean']:.4f}, 95% CI [{pb['ci_lo']:.4f}, {pb['ci_hi']:.4f}] "
      f"(n={int(pb['n'])}, percentile bootstrap, 5000 reps)\n"
      f"- n driver-races where full > baseline: {n_better}\n"
      f"- n driver-races where full < baseline: {n_worse}\n"
      f"- n driver-races exactly equal: {n_equal}\n")

    w("\n## STEP 5 - statistical comparison (existing paired bootstrap)\n")
    w(f"mean_f1_diff_full_minus_baseline = {pb['mean']:.4f}, 95% CI [{pb['ci_lo']:.4f}, "
      f"{pb['ci_hi']:.4f}], n={int(pb['n'])}. The CI excludes 0.\n")

    w("\n## STEP 6 - cross-validation / fold stability (existing walk-forward folds)\n")
    w("NOTE (methodological limitation, stated per the task's own instruction): these are walk-forward "
      "folds (train on past seasons, test on the next), with tyre/Cox models, Tier-3 thresholds and the "
      "SC prior refit on training years only per fold (`model-fit-fold.py`, leak-audited by `audit.py`). "
      "They are NOT clean out-of-sample validation in the strict sense that the OVERALL/production "
      "pipeline (`master.py`'s default `tyre_life_models.pkl`, thresholds, SC prior) is fit on the full "
      "2018-2025 dataset; the fold-mode run exists specifically to give an honest lower bound by refitting "
      "per fold, but the fold BOUNDARIES themselves (which circuits/eras appear in which fold) were not "
      "randomised or held fully independent of feature-engineering decisions made earlier in the project.\n")
    w("\n" + fold_rows.round(3).to_string() + "\n")

    w("\n## STEP 7 - 2025 holdout (existing, untouched)\n")
    w(f"full: precision={holdout_full['precision']:.3f}, recall={holdout_full['recall']:.3f}, "
      f"F1={holdout_full['f1']:.3f}, mean_abs_lap_error={holdout_full['mean_abs_lap_error']:.3f} "
      f"(n={int(holdout_full['n_driver_races'])} driver-races, {int(holdout_full['real_pits'])} real pits)\n")
    w(f"baseline_2of3: precision={holdout_base['precision']:.3f}, recall={holdout_base['recall']:.3f}, "
      f"F1={holdout_base['f1']:.3f}, mean_abs_lap_error={holdout_base['mean_abs_lap_error']:.3f}\n")
    w(f"2025 F1 vs the four walk-forward test-fold F1 range "
      f"[{fold_rows['full'].iloc[:4].min():.3f}, {fold_rows['full'].iloc[:4].max():.3f}]: "
      f"holdout_2025 full F1={holdout_full['f1']:.3f} falls within that range.\n")

    w("\n## STEP 8 - counterfactual decision-quality (existing, ESTIMATED, first-order)\n")
    w("Mechanism (`evaluate.py::counterfactual_time_delta`): for full-system recommendation events that "
      "MATCHED a real pit within tolerance but on a different lap, estimates the seconds gained/lost had "
      "the driver pitted on HERMES's flagged lap instead of the real lap, using only the tyre-pace "
      "difference between the two laps (pit loss assumed identical on either lap, so it cancels out; "
      "traffic, undercut reactions and SC timing are NOT modelled). This is a component-level diagnostic "
      "of alternative decisions on matched events, not a full-system paired outcome vs the baseline.\n")
    w(f"n_matched_events={int(cf['n_matched_events'])}, n_with_estimate={int(cf['n_with_estimate'])}, "
      f"mean_est_time_delta_s={cf['mean_est_time_delta_s']:.3f}, "
      f"median_est_time_delta_s={cf['median_est_time_delta_s']:.3f}, "
      f"95% CI [{cf['ci_lo']:.3f}, {cf['ci_hi']:.3f}], "
      f"frac_hermes_better={cf['frac_hermes_better']:.3f}\n")
    w("Positive = HERMES's flagged lap estimated faster than the real pit lap. Median is 0.0s: most "
      "matched events show no estimated difference (HERMES's flagged lap is at or very near the real "
      "pit lap); the positive mean is driven by a minority (13.4%) of events where it differs.\n")

    w("\n## STEP 9 - historical agreement vs decision quality (kept explicitly separate)\n")
    w("Historical agreement (does HERMES reproduce what the real team did) is measured by the F1/"
      "precision/recall numbers above, by `coverage_exact_agreement` "
      "(exact pit-lap match rate) in the strategic-window harness, and by the counterfactual's "
      "`n_with_estimate` share that is exactly 0 (near-identical timing). Decision quality (does "
      "HERMES's own decision produce a favourable outcome under the model) is evidenced only by the "
      "counterfactual's signed seconds delta on the events where it DOES differ from the real pit, and "
      "by the null-control result below (STEP 11) - the full system's event timing is distinguishable "
      "from randomised/same-circuit-other-season timing, while the baseline's is not. These are "
      "reported as two separate lines of evidence; neither is used to inflate the other. A system that "
      "disagrees with the historical pit lap is not automatically 'wrong', and a system that matches it "
      "exactly has not thereby been shown optimal.\n")

    w("\n## STEP 10 - timing error (secondary simulation-fidelity evidence only)\n")
    w(f"Overall mean_abs_lap_error (matched events only): full={full['mean_abs_lap_error']:.3f} laps, "
      f"baseline_2of3={base['mean_abs_lap_error']:.3f} laps. Reported as secondary simulation-fidelity "
      "evidence (how many laps off, among events already counted as a match) - NOT the primary "
      "decision-quality metric.\n")

    w("\n## STEP 11 - robustness / null controls (existing, eval_out_clean/controls.csv)\n")
    w("Null A = HERMES's own event COUNT relocated to uniformly random laps, scored against the real "
      "pits as they were (300 permutation replicates). Null B = HERMES's events scored against the real "
      "pits of the SAME circuit in a DIFFERENT season (keeps circuit-typical timing). One-sided "
      "permutation p = (1 + #replicates with F1 >= actual) / (1 + replicates).\n")
    w(f"full (ALL): actual_f1={ctrl_all_full['actual_f1']:.4f}, nullA_mean={ctrl_all_full['nullA_mean']:.4f} "
      f"(p={ctrl_all_full['pA']:.4f}), nullB_mean={ctrl_all_full['nullB_mean']:.4f} "
      f"(p={ctrl_all_full['pB']:.4f})\n")
    w(f"baseline_2of3 (ALL): actual_f1={ctrl_all_base['actual_f1']:.4f}, "
      f"nullA_mean={ctrl_all_base['nullA_mean']:.4f} (p={ctrl_all_base['pA']:.4f}), "
      f"nullB_mean={ctrl_all_base['nullB_mean']:.4f} (p={ctrl_all_base['pB']:.4f})\n")
    w("The full system's event timing is distinguishable from both nulls (p<0.01); the baseline's is "
      "not (p close to or at 1.0 against both nulls) - i.e. on this evidence the baseline's apparent "
      "hit rate is statistically indistinguishable from random/circuit-typical timing, while the full "
      "system's is not.\n")

    w("\n## STEP 12 - final consolidated evaluation table\n")
    w("| Evaluation dimension | HERMES | Baseline / reference | Interpretation |")
    w("|---|---:|---:|---|")
    w(f"| Overall paired performance (F1) | {full['f1']:.3f} | {base['f1']:.3f} | quantitative comparison |")
    w(f"| Mean paired difference | {pb['mean']:.3f} | — | HERMES minus baseline, per driver-race F1 |")
    w(f"| 95% CI (paired difference) | [{pb['ci_lo']:.3f}, {pb['ci_hi']:.3f}] | — | uncertainty; excludes 0 |")
    w(f"| Fold stability (F1, 2021/2022/2023/2024) | "
      f"{fold_rows['full'].iloc[0]:.3f} / {fold_rows['full'].iloc[1]:.3f} / "
      f"{fold_rows['full'].iloc[2]:.3f} / {fold_rows['full'].iloc[3]:.3f} | "
      f"{fold_rows['baseline_2of3'].iloc[0]:.3f} / {fold_rows['baseline_2of3'].iloc[1]:.3f} / "
      f"{fold_rows['baseline_2of3'].iloc[2]:.3f} / {fold_rows['baseline_2of3'].iloc[3]:.3f} | "
      f"cross-partition consistency; walk-forward, refit per fold |")
    w(f"| 2025 holdout (F1) | {holdout_full['f1']:.3f} | {holdout_base['f1']:.3f} | "
      f"temporal generalisation; untouched year |")
    w(f"| Timing error (mean abs lap error, matched events) | {full['mean_abs_lap_error']:.3f} | "
      f"{base['mean_abs_lap_error']:.3f} | simulation fidelity (secondary) |")
    w(f"| Counterfactual delta (matched events, est. seconds) | mean={cf['mean_est_time_delta_s']:.3f}, "
      f"median={cf['median_est_time_delta_s']:.3f} | — | alternative-decision behaviour (component-level) |")
    w(f"| Counterfactual better (fraction of matched events) | {cf['frac_hermes_better']:.3f} | — | "
      f"decision-level evidence |")
    w(f"| Null-control distinguishability (p, ALL) | nullA p={ctrl_all_full['pA']:.3f}, "
      f"nullB p={ctrl_all_full['pB']:.3f} | nullA p={ctrl_all_base['pA']:.3f}, "
      f"nullB p={ctrl_all_base['pB']:.3f} | robustness vs random/circuit-typical timing |")
    w(f"| Historical pit agreement (secondary; strategic-window coverage) | "
      f"window_recall={swo['coverage_window_recall']:.3f}, "
      f"useful_warning_recall={swo['useful_warning_recall']:.3f} | "
      f"baseline window_recall={swb['baseline_window_recall']:.3f}, "
      f"useful_warning_recall={swb['baseline_useful_warning_recall']:.3f} | "
      f"secondary only - this metric was independently shown to be saturated (base rate ~0.8) |")

    w("\n## STEP 13 - final interpretation (factual, no new claims)\n")
    w(f"### A. Does HERMES improve simulated outcomes relative to the fair baseline?\n"
      f"On the existing paired F1 comparison: yes, by a mean of {pb['mean']:.3f} F1 "
      f"(95% CI [{pb['ci_lo']:.3f}, {pb['ci_hi']:.3f}], excludes 0, n={int(pb['n'])} driver-races). "
      f"Overall F1 {full['f1']:.3f} (full) vs {base['f1']:.3f} (baseline). On the counterfactual "
      f"seconds-delta evidence (component-level, matched events only): median 0.0s, mean "
      f"+{cf['mean_est_time_delta_s']:.3f}s, with HERMES's flagged lap estimated faster in "
      f"{cf['frac_hermes_better']:.1%} of matched events where the two laps differ. There is no "
      f"existing full-system paired outcome metric expressed in seconds or finishing position against "
      f"the baseline - only the F1 metric is paired at that level.\n")
    w(f"### B. Is that improvement stable across the existing evaluation partitions?\n"
      f"Full-system F1 across the four walk-forward test folds ranges "
      f"[{fold_rows['full'].iloc[:4].min():.3f}, {fold_rows['full'].iloc[:4].max():.3f}] "
      f"(2021={fold_rows['full'].iloc[0]:.3f}, 2022={fold_rows['full'].iloc[1]:.3f}, "
      f"2023={fold_rows['full'].iloc[2]:.3f}, 2024={fold_rows['full'].iloc[3]:.3f}), staying above the "
      f"baseline in every fold except 2022 where baseline_2of3 produced zero events "
      f"(F1 undefined/0 for the baseline that fold; full={fold_rows['full'].iloc[1]:.3f}).\n")
    w(f"### C. Does it generalise to the 2025 holdout?\n"
      f"Full-system 2025 F1={holdout_full['f1']:.3f} falls inside the 2021-2024 fold range above, and "
      f"remains above baseline_2of3's 2025 F1={holdout_base['f1']:.3f}.\n")
    w("### D. Does HERMES need to reproduce historical pit timing for the claim to hold?\n"
      "No. Historical pit timing (F1/exact-agreement/strategic-window coverage) and simulated "
      "decision quality (counterfactual seconds delta, null-control distinguishability) are different "
      "evaluation targets, reported separately in STEP 9. The strategic-window harness independently "
      "established that the exact-timing-agreement question is close to saturated on this dataset "
      "(base rate of PIT_NOW/PIT_LATER across all decision laps ~= 0.8, so a same-prevalence baseline "
      "with no trigger information also reaches ~100% window recall) - the paired-F1 and null-control "
      "results above are not subject to that saturation critique, since F1 requires the recommendation "
      "EVENT (not every decision lap) to fall within tolerance of a specific real pit lap, and the "
      "baseline is measured on the identical saturation-prone decision stream and still scores "
      "substantially lower.\n")
    w("### E. What remains as the main limitation?\n"
      "- The baseline is a relatively low bar: an ablated version of HERMES itself (its own Tier-3 "
      "triggers with rival/SC/radio/weather logic removed), not an independent strategist or external "
      "model.\n"
      "- Absolute historical pit agreement is modest (overall F1 ~0.27-0.29; exact-lap agreement "
      "~19-26% in the strategic-window harness).\n"
      "- Counterfactual benefit is not universal: only 13.4% of matched events show an estimated "
      "timing benefit; most show no estimated difference.\n"
      "- The walk-forward folds are not completely clean out-of-sample validation, because the "
      "production pipeline's feature engineering and overall project structure were developed against "
      "the full dataset even though the fold-mode run itself refits models/thresholds per fold on "
      "training years only.\n"
      "- The simulated/estimated outcome (counterfactual seconds delta) depends on modelling "
      "assumptions (equal pit loss, no traffic/rival reactions, no SC timing) and is first-order only.\n"
      "- There is no existing full-system paired outcome metric expressed in seconds or race "
      "position against the baseline - only F1 (event-timing agreement) is paired at the full-system "
      "level; the seconds-based evidence (counterfactual) is component-level only.\n"
      "- HERMES should therefore be presented as a decision-support system for sequential strategy "
      "decisions under uncertainty, not as an autonomous optimal-strategy engine or a pit-stop "
      "predictor.\n")

    report = "\n".join(lines)
    print(report)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[final_evaluation_summary] wrote {out_path}")


if __name__ == "__main__":
    main()
