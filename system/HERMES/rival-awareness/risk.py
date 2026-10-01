
import pandas as pd

HIGH_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 75

def compute_risk_appetite(signed_points_gap):
    ""
    if signed_points_gap is None or (isinstance(signed_points_gap, float) and pd.isna(signed_points_gap)):
        return None
    abs_gap = abs(signed_points_gap)
    if abs_gap <= HIGH_RISK_THRESHOLD:
        return "high"
    elif abs_gap <= MEDIUM_RISK_THRESHOLD:
        return "medium"
    else:
        return "low"
