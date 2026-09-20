"""
Risk Policy — Championship Pressure -> Risk Appetite
=======================================================
Deliberately simple and rule-based, not a trained model — per the explicit design
choice to keep this "transparent and explainable" rather than opaque. This is the
piece that moves the "incorporate championship standings weight" checklist item from
PARTIAL (a points_gap column just sitting there) to DONE (the gap actually feeding a
named, documented policy that a downstream strategy system could act on).

POLICY: a driver's willingness to take strategic risk against a SPECIFIC rival is
modeled as a function of how much the championship gap to THAT rival matters right now
— not overall championship position, since a big points gap to a rival you're not
actually racing for position doesn't change how you should treat undercutting them.

  HIGH risk appetite   -> |signed_gap| <= 25 points
      Roughly one race win/podium swing. The battle with this specific rival is
      genuinely live — both drivers have strong incentive to fight hard for track
      position, since the gap could flip in a single race.

  MEDIUM risk appetite -> 25 < |signed_gap| <= 75 points
      A gap that matters over a season but isn't decided by any single race.

  LOW risk appetite    -> |signed_gap| > 75 points
      Large enough that a single track-position swing against this rival is unlikely
      to matter much — conservative racing (protect the current result, avoid a risky
      strategic gamble) is more rational than fighting hard for a position that barely
      moves the championship needle.

THRESHOLDS ARE A STARTING HEURISTIC, NOT FITTED OR VALIDATED against real outcomes —
documented explicitly as provisional, matching how every other threshold in this
project has been handled (MIN_CLEAN_LAPS_FOR_BASELINE, MAX_PLAUSIBLE_PIT_STOP_S, the
8-35s pit-loss plausibility band). 25/75 points were chosen because they roughly
correspond to 1 and 3 race wins respectively under the current points system — a
reasonable starting point for a dissertation-stage feature, worth revisiting with
real validation (e.g. checking whether teams' actual reported strategic aggression
correlates with these bands) before using it for anything beyond demonstrating that
the feature is wired through end to end.
"""

import pandas as pd

HIGH_RISK_THRESHOLD = 25    # points
MEDIUM_RISK_THRESHOLD = 75  # points


def compute_risk_appetite(signed_points_gap):
    """signed_points_gap: flagged driver's points minus rival's points, BEFORE the
    race (see standings.get_signed_points_gap). Direction doesn't matter for this
    policy — being 100 points ahead or 100 points behind a rival both mean the gap
    to THAT rival is settled enough that risk appetite should be low."""
    if signed_points_gap is None or (isinstance(signed_points_gap, float) and pd.isna(signed_points_gap)):
        return None
    abs_gap = abs(signed_points_gap)
    if abs_gap <= HIGH_RISK_THRESHOLD:
        return "high"
    elif abs_gap <= MEDIUM_RISK_THRESHOLD:
        return "medium"
    else:
        return "low"