import pandas as pd
import re

df = pd.read_csv("data\\external\\team-radios\\transcriptions.csv")

CATEGORIES_DICT = {
    "pit_instructions": ["box this lap", "box box", "box", "box, box", "box now", "stay out", "boxing this lap",],
    "tyre_feedback":,
    "pace_management":,
    "safety_Car_vsc":,
    "position_gap_info":,
    "team_order":,
    "mechanical_issue":,
    "acknowledgement":,
    "other":
}