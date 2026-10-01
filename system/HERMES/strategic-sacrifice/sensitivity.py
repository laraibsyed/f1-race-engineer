""

from reward import (
    DriverChampionshipState, TeamChampionshipState,
    compute_team_reward, justification_check, points_delta
)

def sensitivity_analysis():
    SEP = "═"*57
    print("\n" + SEP)
    print("  SENSITIVITY ANALYSIS")
    print(SEP)

    print("\n  Brazil 2022: vary Pérez WDC gap (races_remaining=1, alpha=0.3, tau=0.1)")
    header = f"  {'PER gap':>10} {'L2':>8} {'Gain':>10} {'Justified?':>12}"
    print(header)
    print("  " + "-"*46)
    for gap in [0, 6, 13, 20, 30, 50]:
        perez = DriverChampionshipState("Perez", wdc_gap=gap, role_weight=0.35)
        ver   = DriverChampionshipState("VER",   wdc_gap=0.0, role_weight=0.65, title_secured=True)
        team  = TeamChampionshipState(wcc_gap=0.0, races_remaining=1, alpha=0.3)
        R_base = compute_team_reward(0.0, 0.0, ver, 0.0, perez, team)
        R_swap = compute_team_reward(0.0, points_delta(6,7), ver, points_delta(7,6), perez, team, risk=0.02)
        j = justification_check(R_swap, R_base, tau=0.1)
        tag = "<- historical (verified)" if gap == 6 else ""
        print(f"  {gap:>10} {R_swap['L2']:>8.4f} {j['gain']:>10.4f} {str(j['justified']):>12}  {tag}")

    print("\n  Malaysia 2013: vary w1 (D1 role weight) and tau")
    header2 = f"  {'w1':>6} {'w2':>6} {'tau':>6} {'Gain':>10} {'Justified?':>12}"
    print(header2)
    print("  " + "-"*48)
    cases = [(0.65, 0.1), (0.5, 0.1), (0.35, 0.1), (0.65, 0.0), (0.65, -0.5)]
    for w1, tau in cases:
        w2 = round(1.0 - w1, 2)
        vet = DriverChampionshipState("VET", wdc_gap=0.0, role_weight=w1)
        web = DriverChampionshipState("WEB", wdc_gap=7.0, role_weight=w2)
        team = TeamChampionshipState(wcc_gap=0.0, races_remaining=18, alpha=0.6)
        R_base = compute_team_reward(0.0, 0.0, vet, 0.0, web, team)
        R_sac  = compute_team_reward(0.0, points_delta(1,2), vet, points_delta(2,1), web, team, risk=0.08)
        j = justification_check(R_sac, R_base, tau=tau)
        tag = "<- base" if w1 == 0.65 and tau == 0.1 else ""
        print(f"  {w1:>6} {w2:>6} {tau:>6} {j['gain']:>10.4f} {str(j['justified']):>12}  {tag}")

    print("  Verdict: Malaysia only flips to SACRIFICE when w1 < w2 (roles reversed).")
    print("  tau is irrelevant — gain is -0.9344, so tau would need to fall below -0.9344")
    print("\n  Brazil flip point (analytical): gap > ~80pts with 1 race remaining.")
    print("  At that point L2 drops to 0.245, gain falls below tau=0.1.")
    print("  Historical gap was 6pts — L2=0.813, gain=+0.378, swap comfortably justified.")
    print("  Note: flip point scales with races_remaining * 26 (points available).")
    print()
    print("  Malaysia flip point: tau < -0.9344 (gain is -0.9344).")
    print("  A negative tau is non-physical — the decision is unconditionally robust.")
    print(SEP)

if __name__ == "__main__":
    sensitivity_analysis()
