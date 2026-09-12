"""
Survival Model V1 — Kaplan-Meier (no covariates)
====================================================
Descriptive per-compound survival curves: "what's the probability this tyre
hasn't hit its cliff by tyre_age N?" No covariates yet - that's V2 (Cox
proportional hazards with track_temp_bucket, fuel_load_estimate, stint_number,
Driver - same covariate list flagged for the regression V2 work).

Input: cliff_detection_stints.csv (produced by cliff_detection.py) - duration/event
columns are already in the correct shape for survival analysis, so this reads the
local CSV directly rather than going back to GCS.

Produces:
  - survival_v1_km_curves.png   (overlaid KM curves per compound)
  - console: median survival (tyre_age) per compound + log-rank test across compounds
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test

STINTS_CSV = "cliff_detection_stints.csv"
MIN_STINTS_PER_COMPOUND = 15  # skip compounds with too few stints for a trustworthy curve


if __name__ == "__main__":
    stints = pd.read_csv(STINTS_CSV)
    n_rbr = stints["is_rbr"].sum() if "is_rbr" in stints.columns else None
    print(f"[load] {len(stints)} stints from {STINTS_CSV}"
          + (f" ({n_rbr} RBR)" if n_rbr is not None else ""))
    print("[scope] KM curves below are fit on ALL TEAMS for statistical power - "
          "see cliff_detection.py's load_all_teams_laps docstring for the reasoning")

    compound_counts = stints["compound"].value_counts()
    compounds_to_plot = compound_counts[compound_counts >= MIN_STINTS_PER_COMPOUND].index.tolist()
    skipped = compound_counts[compound_counts < MIN_STINTS_PER_COMPOUND].index.tolist()
    print(f"[info] plotting: {compounds_to_plot}")
    if skipped:
        print(f"[info] skipped (too few stints, <{MIN_STINTS_PER_COMPOUND}): {skipped}")

    fig, ax = plt.subplots(figsize=(9, 7))
    median_survival = {}
    fitted_kmfs = []

    for compound in compounds_to_plot:
        g = stints[stints["compound"] == compound]
        kmf = KaplanMeierFitter(label=compound)
        kmf.fit(durations=g["duration"], event_observed=g["event"])
        kmf.plot_survival_function(ax=ax)
        median_survival[compound] = kmf.median_survival_time_
        fitted_kmfs.append(kmf)

    add_at_risk_counts(*fitted_kmfs, ax=ax)  # shows how thin the sample gets at high tyre_age -
                                              # read anything past where these counts get small
                                              # (especially MEDIUM/HARD's tail) with real caution

    ax.set_xlabel("tyre_age (laps)")
    ax.set_ylabel("Probability tyre hasn't hit cliff yet")
    ax.set_title("Kaplan-Meier survival curves by compound (RBR, V1 — no covariates)")
    ax.grid(alpha=0.3)
    plt.savefig("survival_v1_km_curves.png", dpi=150, bbox_inches="tight")
    print("\n[save] survival_v1_km_curves.png")

    print("\n=== MEDIAN SURVIVAL (tyre_age at which 50% of stints have hit a cliff) ===")
    for compound, median in sorted(median_survival.items(), key=lambda kv: kv[1]):
        label = f"{median:.0f} laps" if np.isfinite(median) else "not reached (>50% still censored at max observed tyre age)"
        print(f"  {compound:>14}: {label}")

    # Log-rank test: are the survival curves statistically different across compounds?
    tested = stints[stints["compound"].isin(compounds_to_plot)]
    result = multivariate_logrank_test(tested["duration"], tested["compound"], tested["event"])
    print(f"\n[log-rank test] p-value = {result.p_value:.4f} "
          f"({'compounds ARE significantly different' if result.p_value < 0.05 else 'no significant difference detected'})")