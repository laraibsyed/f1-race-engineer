
from reward import (
    DriverChampionshipState, TeamChampionshipState,
    compute_team_reward, justification_check, points_delta
)

def input_audit(label, **kwargs):
    print(f"\n  ┌─ INPUT AUDIT: {label}")
    for k, v in kwargs.items():
        print(f"  │  {k:<36}: {v}")
    print(f"  └{'─'*46}")

def run_russia_2018():
    print("\n" + "╔" + "═"*55 + "╗")
    print("║  CASE 3: Russia 2018 — Bottas yields to Hamilton   ║")
    print("╚" + "═"*55 + "╝")

    input_audit("Russia 2018 — verified historical inputs",
        source_standings="espn.in pre-race preview / crash.net post-race",
        source_result="gparchive.com official result / formula1.com team quotes",
        round="16 of 21",
        hamilton_pts_entering_race=281,
        vettel_pts_entering_race=244,
        hamilton_gap_to_vettel="37 pts ahead (HAM leads)",
        bottas_pts_entering_race=171,
        bottas_gap_to_hamilton="110 pts behind — title mathematically unlikely",
        races_remaining=5,
        points_available="5 × 26 = 130 pts",
        wcc_mercedes_pre_race=459,
        wcc_ferrari_pre_race=418,
        wcc_gap="41 pts — Mercedes leads, not yet secured",
        decision_point="Lap ~26/53, BOT P1 leads HAM P2 by ~0.9s",
        swap_delta="HAM P2→P1 (+7 pts), BOT P1→P2 (−7 pts)",
        race_result="P1 HAM (25), P2 BOT (18), P3 VET (15), P4 RAI (4)",
        risk_estimate="0.01 — clean swap, Bottas complied immediately",
        tau="0.1 — consistent with other cases"
    )

    hamilton = DriverChampionshipState(
        name="Hamilton",
        wdc_gap=0.0,
        role_weight=0.65,
        title_secured=False
    )

    bottas = DriverChampionshipState(
        name="Bottas",
        wdc_gap=110.0,
        role_weight=0.35,
        title_secured=False
    )
    team = TeamChampionshipState(
        wcc_gap=-41.0,
        races_remaining=5,
        alpha=0.45
    )
    tau = 0.1

    hamilton = DriverChampionshipState(
        name="Hamilton",
        wdc_gap=37.0,
        role_weight=0.65,
        title_secured=False
    )

    print("\n── Baseline: No swap — BOT P1 (25pts), HAM P2 (18pts)")
    R_baseline = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=0.0,
        d1_state=hamilton,
        delta_wdc_d2=0.0,
        d2_state=bottas,
        team_state=team,
        risk=0.0,
        verbose=True
    )

    print("── Sacrifice: Swap — HAM P1 (25pts), BOT P2 (18pts)")
    R_swap = compute_team_reward(
        delta_wcc=0.0,
        delta_wdc_d1=points_delta(2, 1),
        d1_state=hamilton,
        delta_wdc_d2=points_delta(1, 2),
        d2_state=bottas,
        team_state=team,
        risk=0.01,
        verbose=True
    )

    result = justification_check(R_swap, R_baseline, tau=tau, verbose=True)

    print(f"  Explanation:")
    print(f"  HAM L1 = 1/(1 + 37/130) = 0.778 — title fight live, +7pts meaningful.")
    print(f"  BOT L2 = 1/(1 + 110/130) = 0.542 — 110pts down, fight near-dead.")
    print(f"  BOT's −7pt loss is heavily discounted by low leverage.")
    print(f"  Net WDC component positive → swap justified by WDC asymmetry.")
    print(f"  WCC delta = 0; WCC weight moderate (α=0.45, not yet secured).")
    print(f"  Expected: SACRIFICE | Got: {result['recommendation']}")
    print(f"  Match: {'✓ CORRECT' if result['recommendation'] == 'SACRIFICE' else '✗ WRONG'}")
    print(f"\n  Historical outcome: Bottas complied on lap 26. Hamilton won.")
    print(f"  Post-race gap Hamilton→Vettel: 50pts (up from 37pts).")
    return result

if __name__ == "__main__":
    run_russia_2018()
