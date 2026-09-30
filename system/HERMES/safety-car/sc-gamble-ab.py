r"""
SC Gamble A/B: analytic evaluate_sc_gamble  vs  Monte Carlo evaluate_sc_gamble_mc
======================================================================================
Runs the SAME parameter grids sc-gamble-v2.py's run_sweep() used (full 11x11x11
grid, and the "realistic operating range" grid, both at horizon 5) through both
evaluators and reports whether the Monte Carlo version is a faithful
uncertainty-propagation wrapper around the analytic reference - NOT whether it
is "more right". Framing (deliberate): the analytic model is the validated
reference; MC quantifies uncertainty around the same cost model and must not
change the strategy logic. Passing means the two agree on decisions everywhere
except cells that are genuine toss-ups within simulation noise.

Checks:
  1. schema        MC return keys are a superset of the analytic keys; same vocabulary
  2. semantics     INSUFFICIENT_DATA parity for missing inputs
  3. reproducible  same seed -> identical output
  4. agreement     decision agreement over both grids; every disagreement classified as
                   "within MC noise" (|analytic saving| < 3 x MC standard error) or "MATERIAL"
  5. seed stability  decisions that flip across seeds must be the same noise-level cells
  6. mean fidelity   MC mean saving vs analytic mean saving

Usage (from the repo dir):
    .\.venv\Scripts\python.exe system\HERMES\safety-car\sc-gamble-ab.py
Exit code 0 = all checks passed.
"""
import importlib.util
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sc_gamble", HERE / "sc-gamble.py")
scg = importlib.util.module_from_spec(spec)
sys.modules["sc_gamble"] = scg                    # required by dataclasses on Python 3.14
spec.loader.exec_module(scg)

S = scg.SCGambleInputs
VOCAB = {"WAIT", "NO_ADVANTAGE_TO_WAITING", "INSUFFICIENT_DATA"}
HORIZON = 5
SEEDS = (0, 1, 2, 3)
NOISE_SE = 3.0   # a cell is a "toss-up" if |analytic saving| < NOISE_SE x MC standard error. 3 (not 2): with hundreds of
                 # cells scanned, some sit right at 2 sigma and WILL flip on a different seed - that is expected sampling
                 # noise, not a defect. Raising n_sims shrinks the band.

FULL_GRID = list(itertools.product(np.linspace(0, 1, 11), np.linspace(0, 1, 11), np.linspace(0, 1, 11)))
REAL_GRID = list(itertools.product(np.linspace(0, 0.15, 8), np.linspace(0, 0.10, 6), np.linspace(0, 0.30, 7)))
# tuple order in both grids: (p_sc, cliff_probability, pace_loss_per_lap)

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def run_grid(name, grid):
    print(f"\n=== {name}: {len(grid)} cells, horizon {HORIZON} ===")
    rows = []
    for p, c, pace in grid:
        inp = S(p, pace, c, HORIZON)
        a = scg.evaluate_sc_gamble(inp)
        m = scg.evaluate_sc_gamble_mc(inp, scg.MCConfig(seed=SEEDS[0]))
        rows.append((p, c, pace, a, m))

    agree = sum(a["recommendation"] == m["recommendation"] for *_, a, m in rows)
    print(f"  analytic WAIT cells: {sum(a['recommendation'] == 'WAIT' for *_, a, m in rows)}   "
          f"MC WAIT cells: {sum(m['recommendation'] == 'WAIT' for *_, a, m in rows)}")
    print(f"  decision agreement: {agree}/{len(rows)} ({agree / len(rows):.1%})")

    disagree = [(p, c, pace, a, m) for p, c, pace, a, m in rows if a["recommendation"] != m["recommendation"]]
    material = [r for r in disagree if abs(r[3]["expected_saving_if_wait"]) >= NOISE_SE * r[4]["saving_se"]]
    noise = len(disagree) - len(material)
    print(f"  disagreements: {len(disagree)}  ->  within MC noise (toss-up): {noise}   MATERIAL: {len(material)}")
    for p, c, pace, a, m in material[:8]:
        print(f"     MATERIAL p_sc={p:.2f} cliff={c:.2f} pace={pace:.2f}: analytic {a['expected_saving_if_wait']:+.3f}s "
              f"({a['recommendation']}) vs MC {m['expected_saving_if_wait']:+.3f}s ({m['recommendation']}), se {m['saving_se']:.3f}")
    check(f"{name}: no MATERIAL disagreement", len(material) == 0)

    diffs = np.array([m["expected_saving_if_wait"] - a["expected_saving_if_wait"] for *_, a, m in rows])
    print(f"  MC - analytic mean saving: mean {diffs.mean():+.3f}s, max |diff| {np.abs(diffs).max():.3f}s, "
          f"95th pct |diff| {np.percentile(np.abs(diffs), 95):.3f}s")
    check(f"{name}: MC mean within 0.25s of analytic in 95% of cells", np.percentile(np.abs(diffs), 95) <= 0.25)

    # seed stability: which cells flip when only the seed changes?
    base = {(p, c, pace): m["recommendation"] for p, c, pace, a, m in rows}
    flips = set()
    for seed in SEEDS[1:]:
        for p, c, pace in grid:
            r = scg.evaluate_sc_gamble_mc(S(p, pace, c, HORIZON), scg.MCConfig(seed=seed))
            if r["recommendation"] != base[(p, c, pace)]:
                flips.add((p, c, pace))
    by_key = {(p, c, pace): (a, m) for p, c, pace, a, m in rows}
    non_noise_flips = [k for k in flips if abs(by_key[k][0]["expected_saving_if_wait"]) >= NOISE_SE * by_key[k][1]["saving_se"]]
    if flips:
        mags = [abs(by_key[k][0]["expected_saving_if_wait"]) for k in flips]
        print(f"  cells whose decision changes with the seed: {len(flips)} (of which outside the {NOISE_SE:g}-sigma noise band: "
              f"{len(non_noise_flips)}); largest |analytic saving| among them: {max(mags):.3f}s")
    else:
        print("  cells whose decision changes with the seed: 0")
    check(f"{name}: seed-sensitive cells are all toss-ups", len(non_noise_flips) == 0)
    return rows


print("=" * 78)
print("1-3. schema, semantics, reproducibility")
print("=" * 78)
probe = S(0.15, 0.1, 0.02, HORIZON)
a, m = scg.evaluate_sc_gamble(probe), scg.evaluate_sc_gamble_mc(probe)
check("MC keys are a superset of analytic keys", set(a) <= set(m), f"extra: {sorted(set(m) - set(a))}")
check("recommendations are in the shared vocabulary", a["recommendation"] in VOCAB and m["recommendation"] in VOCAB)
check("cost fields are floats", all(isinstance(m[k], float) for k in ("cost_pit_now", "cost_wait", "expected_saving_if_wait")))
check("expected_saving == cost_pit_now - cost_wait (MC)", abs(m["expected_saving_if_wait"] - (m["cost_pit_now"] - m["cost_wait"])) < 1e-9)
for label, bad in [("p_sc None", S(None, 0.1, 0.0, HORIZON)), ("pace None", S(0.1, None, 0.0, HORIZON))]:
    ra, rm = scg.evaluate_sc_gamble(bad), scg.evaluate_sc_gamble_mc(bad)
    check(f"INSUFFICIENT_DATA parity: {label}", ra["recommendation"] == rm["recommendation"] == "INSUFFICIENT_DATA"
          and rm["cost_pit_now"] is None and rm["cost_wait"] is None)
check("cliff None treated as 0 (both evaluators run)", scg.evaluate_sc_gamble(S(0.1, 0.1, None, HORIZON))["recommendation"] in VOCAB
      and scg.evaluate_sc_gamble_mc(S(0.1, 0.1, None, HORIZON))["recommendation"] in VOCAB)
r1, r2 = scg.evaluate_sc_gamble_mc(probe, scg.MCConfig(seed=7)), scg.evaluate_sc_gamble_mc(probe, scg.MCConfig(seed=7))
check("same seed -> identical output", r1 == r2)
check("default config is deterministic run-to-run", scg.evaluate_sc_gamble_mc(probe) == scg.evaluate_sc_gamble_mc(probe))
check("uses the empirical pit-stop pools", str(m["pit_duration_source"]).startswith("empirical"), m["pit_duration_source"])

print("\n" + "=" * 78)
print("4-6. decision agreement on sc-gamble-v2.py's own grids")
print("=" * 78)
run_grid("FULL grid (p_sc x cliff x pace, 0-1)", FULL_GRID)
real_rows = run_grid("REALISTIC range (p_sc<=0.15, cliff<=0.10, pace<=0.30)", REAL_GRID)

print("\n=== Where the two versions carry different information (not disagreement) ===")
conf = Counter(m["decision_confident"] for *_, a, m in real_rows)
pw = np.array([m["p_wait_better"] for *_, a, m in real_rows])
print(f"  realistic grid: decision_confident True/False = {conf.get(True, 0)}/{conf.get(False, 0)}")
print(f"  P(waiting beats pitting now) across the realistic grid: min {pw.min():.1%}, median {np.median(pw):.1%}, max {pw.max():.1%}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
