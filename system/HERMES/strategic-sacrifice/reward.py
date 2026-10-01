
from dataclasses import dataclass
from typing import Optional

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
             6: 8,  7: 6,  8: 4,  9: 2,  10: 1}

def position_points(pos: int) -> float:
    ""
    return F1_POINTS.get(pos, 0.0)

def points_delta(pos_before: int, pos_after: int) -> float:
    ""
    return position_points(pos_after) - position_points(pos_before)

@dataclass
class DriverChampionshipState:
    ""
    name: str
    wdc_gap: float
    role_weight: float
    title_secured: bool = False

@dataclass
class TeamChampionshipState:
    ""
    wcc_gap: float
    races_remaining: int
    alpha: float = 0.5

def championship_leverage(gap: float, races_remaining: int,
                          title_secured: bool = False) -> float:
    ""
    if title_secured:
        return 0.0
    points_available = races_remaining * 26
    if points_available <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(gap) / points_available)

def compute_team_reward(

    delta_wcc: float,

    delta_wdc_d1: float,
    d1_state: DriverChampionshipState,

    delta_wdc_d2: float,
    d2_state: DriverChampionshipState,

    team_state: TeamChampionshipState,

    risk: float = 0.0,
    lambda_risk: float = 1.0,
    verbose: bool = False
) -> dict:
    ""
    alpha = team_state.alpha

    L1 = championship_leverage(
        d1_state.wdc_gap, team_state.races_remaining, d1_state.title_secured)
    L2 = championship_leverage(
        d2_state.wdc_gap, team_state.races_remaining, d2_state.title_secured)

    wcc_component = alpha * delta_wcc

    wdc_d1 = d1_state.role_weight * L1 * delta_wdc_d1
    wdc_d2 = d2_state.role_weight * L2 * delta_wdc_d2
    wdc_component = (1 - alpha) * (wdc_d1 + wdc_d2)

    risk_penalty = lambda_risk * risk

    R_team = wcc_component + wdc_component - risk_penalty

    result = {
        "R_team": round(R_team, 4),
        "wcc_component": round(wcc_component, 4),
        "wdc_component": round(wdc_component, 4),
        "wdc_d1_term": round(wdc_d1, 4),
        "wdc_d2_term": round(wdc_d2, 4),
        "risk_penalty": round(risk_penalty, 4),
        "L1": round(L1, 4),
        "L2": round(L2, 4),
        "alpha": alpha,
        "w1": d1_state.role_weight,
        "w2": d2_state.role_weight,
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  R_team Breakdown")
        print(f"{'─'*55}")
        print(f"  α (WCC weight)          : {alpha}")
        print(f"  WCC component  α·ΔWCC   : {wcc_component:+.4f}")
        print(f"  L1 (D1 leverage)        : {L1:.4f}")
        print(f"  L2 (D2 leverage)        : {L2:.4f}")
        print(f"  WDC D1 term w1·L1·ΔWDC1: {wdc_d1:+.4f}")
        print(f"  WDC D2 term w2·L2·ΔWDC2: {wdc_d2:+.4f}")
        print(f"  WDC component           : {wdc_component:+.4f}")
        print(f"  Risk penalty  λ·risk    : {risk_penalty:+.4f}")
        print(f"{'─'*55}")
        print(f"  R_team                  : {R_team:+.4f}")
        print(f"{'─'*55}\n")

    return result

def justification_check(
    R_sacrifice: dict,
    R_no_sacrifice: dict,
    tau: float = 0.5,
    verbose: bool = False
) -> dict:
    ""
    gain = R_sacrifice["R_team"] - R_no_sacrifice["R_team"]
    justified = gain > tau

    result = {
        "R_sacrifice":    round(R_sacrifice["R_team"], 4),
        "R_no_sacrifice": round(R_no_sacrifice["R_team"], 4),
        "gain":           round(gain, 4),
        "tau":            tau,
        "justified":      justified,
        "recommendation": "SACRIFICE" if justified else "HOLD POSITIONS",
    }

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Justification Check")
        print(f"{'═'*55}")
        print(f"  R_team | sacrifice     : {R_sacrifice['R_team']:+.4f}")
        print(f"  R_team | no sacrifice  : {R_no_sacrifice['R_team']:+.4f}")
        print(f"  Gain                   : {gain:+.4f}")
        print(f"  Compliance cost τ      : {tau}")
        print(f"  Justified?             : {justified}")
        print(f"  → Recommendation       : {result['recommendation']}")
        print(f"{'═'*55}\n")

    return result
