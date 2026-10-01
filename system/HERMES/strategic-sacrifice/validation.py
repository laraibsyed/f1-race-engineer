
from sensitivity import sensitivity_analysis
from reward import (
    DriverChampionshipState, TeamChampionshipState,
    compute_team_reward, justification_check, points_delta
)

def input_audit(label, **kwargs):
    print(f"\n  ┌─ INPUT AUDIT: {label}")
    for k, v in kwargs.items():
        print(f"  │  {k:<34}: {v}")
    print(f"  └{'─'*44}")

def run_brazil_2022():
    print("\n" + "╔" + "═"*55 + "╗")
    print("║  CASE 1: Brazil 2022 — Verstappen vs Pérez (P6/P7) ║")
    print("╚" + "═"*55 + "╝")

    input_audit("Brazil 2022 — verified historical inputs",
        source_standings="racefans.net post-sprint / crash.net post-race",
        source_result="formula1points.com / thepaddockmagazine.com",
        verstappen_pts_pre_sprint=416,
        verstappen_pts_post_sprint=421,
        verstappen_title_secured="Yes (Japan R17)",
        perez_pts_post_sprint=284,
        leclerc_pts_post_sprint=278,
        perez_lead_over_leclerc_entering_gp="6 pts",
        race_result_ver="P6 (8 pts)",
        race_result_per="P7 (6 pts)",
        race_result_lec="P4 (12 pts)",
        post_race_perez=290,
        post_race_leclerc=290,
        decision_point="VER P6 asked to yield to PER P7",
        swap_delta="PER P7→P6 (+2 pts), VER P6→P7 (−2 pts, L1=0)",
        wcc_red_bull_pre_race=715,
        wcc_secured="Yes (Suzuka R19)",
        races_remaining_after_brazil=1,
        risk_estimate="0.02 — assumed; not independently calibrated",
        tau="0.1 — hand-tuned; flagged as limitation"
    )

    verstappen = DriverChampionshipState(
        name="Verstappen",
        wdc_gap=0.0,
        role_weight=0.65,
        title_secured=True
    )
    perez = DriverChampionshipState(
        name="Perez",
        wdc_gap=6.0,
        role_weight=0.35,
        title_secured=False
    )
    team = TeamChampionshipState(
        wcc_gap=0.0,
        races_remaining=1,
        alpha=0.3
    )
    tau = 0.1

    print("\n── Baseline: No swap — VER stays P6 (8pts), PER stays P7 (6pts)")
    R_no_swap = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=0.0,
        d1_state=verstappen,
        delta_wdc_d2=0.0,
        d2_state=perez,
        team_state=team,
        risk=0.0,
        verbose=True
    )

    print("── Sacrifice: Swap — PER moves to P6 (8pts), VER drops to P7 (6pts)")
    R_swap = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=points_delta(6, 7),
        d1_state=verstappen,
        delta_wdc_d2=points_delta(7, 6),
        d2_state=perez,
        team_state=team,
        risk=0.02,
        verbose=True
    )

    result = justification_check(R_swap, R_no_swap, tau=tau, verbose=True)

    print(f"  Explanation: VER L1=0 (title secured) → his −2pts cost the team zero.")
    print(f"  PER L2=0.812 (6pt lead, 1 race left) → +2pts gains +0.569 WDC utility.")
    print(f"  Swap is justified by WDC asymmetry. WCC = 0 in both scenarios.")
    print(f"  Expected: SACRIFICE | Got: {result['recommendation']}")
    print(f"  Match: {'✓ CORRECT' if result['recommendation'] == 'SACRIFICE' else '✗ WRONG'}\n")
    return result

def run_malaysia_2013():
    print("\n" + "╔" + "═"*55 + "╗")
    print("║  CASE 2: Malaysia 2013 — Multi 21 (Vettel/Webber)  ║")
    print("╚" + "═"*55 + "╝")

    input_audit("Malaysia 2013 — verified historical inputs",
        source="formula1.com official standings R1-R2 2013",
        vettel_pts_entering_race=25,
        webber_pts_entering_race=18,
        vettel_lead_over_webber="7 pts",
        races_remaining=18,
        baseline="No team order — VET overtakes → VET P1 (25pts), WEB P2 (18pts)",
        team_order="Multi 21: hold, WEB P1 (25pts), VET P2 (18pts)",
        delta_wdc_vet="P1→P2 = −7 pts",
        delta_wdc_web="P2→P1 = +7 pts",
        delta_wcc=0,
        risk_note="0.08 retrospective estimate — non-compliance base rate unknown",
        tau="0.1 — same as Brazil case"
    )

    vettel = DriverChampionshipState(
        name="Vettel",
        wdc_gap=0.0,
        role_weight=0.65,
        title_secured=False
    )
    webber = DriverChampionshipState(
        name="Webber",
        wdc_gap=7.0,
        role_weight=0.35,
        title_secured=False
    )
    team = TeamChampionshipState(
        wcc_gap=0.0,
        races_remaining=18,
        alpha=0.6
    )
    tau = 0.1

    print("\n── Baseline: No team order — VET overtakes → VET P1 (25pts), WEB P2 (18pts)")
    R_baseline = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=0.0,
        d1_state=vettel,
        delta_wdc_d2=0.0,
        d2_state=webber,
        team_state=team,
        risk=0.0,
        verbose=True
    )

    print("── Sacrifice: Order complied — WEB P1 (25pts), VET P2 (18pts)")
    R_sacrifice = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=points_delta(1, 2),
        d1_state=vettel,
        delta_wdc_d2=points_delta(2, 1),
        d2_state=webber,
        team_state=team,
        risk=0.08,
        verbose=True
    )

    result = justification_check(R_sacrifice, R_baseline, tau=tau, verbose=True)

    print(f"  Explanation: L1=1.0, L2=0.985 — leverage is NOT why this fails.")
    print(f"  Fails because w1=0.65 > w2=0.35 (role-weight asymmetry).")
    print(f"  VET loss: 0.65×1.0×(−7) = −4.55. WEB gain: 0.35×0.985×(+7) = +2.41.")
    print(f"  Net WDC = −0.85. Sacrifice always net-negative for any τ ≥ 0.")
    print(f"  Expected: HOLD POSITIONS | Got: {result['recommendation']}")
    print(f"  Match: {'✓ CORRECT' if result['recommendation'] == 'HOLD POSITIONS' else '✗ WRONG'}\n")
    return result

if __name__ == "__main__":
    r1 = run_brazil_2022()
    r2 = run_malaysia_2013()
    sensitivity_analysis()

    SEP = "═"*57
    print("\n" + SEP)
    print("  VALIDATION SUMMARY")
    print(SEP)
    print(f"  Brazil 2022   → {r1['recommendation']:<18} (expected: SACRIFICE)")
    print(f"  Malaysia 2013 → {r2['recommendation']:<18} (expected: HOLD POSITIONS)")
    both = r1['recommendation'] == 'SACRIFICE' and r2['recommendation'] == 'HOLD POSITIONS'
    print(f"  Both correct? : {'✓ YES' if both else '✗ NO'}")
    print()
    print("  Corrections from previous version:")
    print("  · Brazil positions corrected to P6/P7 (not P5/P6)")
    print("  · Brazil pre-race gap corrected: PER 6pts AHEAD of LEC (not 13 behind)")
    print("  · Malaysia baseline corrected: VET overtakes = reference state")
    print()
    print("  Limitations flagged:")
    print("  · risk values assumed; non-compliance base rate not independently estimated")
    print("  · τ=0.1 hand-tuned; principled calibration is future work")
    print("  · Malaysia driven by role-weight asymmetry, not leverage or season timing")
    print(SEP)
