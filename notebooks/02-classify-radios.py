import pandas as pd
import re

df = pd.read_csv("data\\external\\team-radios\\transcriptions.csv")

CATEGORIES_DICT = {
    "pit_instruction": [
        "box", "stay out", "come in", "pit this lap", "pit now",
        "stack", "speed limiter", "pit window", "undercut", "overcut",
        "we are boxing", "retiring the car", "in this lap", "box for"
    ],
    "tyre_feedback": [
        "tyre", "tire", "grip", "degradation", "deg", "wear",
        "graining", "blistering", "front left", "front right",
        "rear left", "rear right", "compound", "soft", "medium",
        "hard", "inter", "intermediate", "wet tyre", "prime", "option",
        "stint", "no grip", "gone", "dropping off", "losing the rear",
        "rear is gone", "front is gone", "flat spot", "hot", "overheating tyre", 
        "rears", "fronts", "rear temp", "front temp"
    ],
    "pace_management": [
        "push now", "push hard", "back off", "save",
        "conserve", "manage", "lift and coast", "lift", "coast",
        "target lap", "delta", "gap", "offset", "fuel", "fuel save",
        "engine mode", "strat", "battery", "deploy",
        "harvest", "cool down lap", "rain", 
        "weather", "wind", "last lap was", "lap time", "overtake available",
        "target lap time", "push now", "push hard", "push push",
        "fuel save", "save fuel", "lift and coast", "mode sc",
        "strat", "energy", "deploy", "harvest", "conserve",
        "back off", "manage the gap", "cool the tyres"
    ],
    "safety_car_vsc": [
        "safety car", "virtual safety car", "vsc", "sc", "delta",
        "positive on the delta", "yellow", "yellow flag", "red flag",
        "neutralised", "restart", "formation lap", "safety car in",
        "safety car out", "safety car this lap", "deployed"
    ],
    "position_gap_info": [
        "gap", "seconds behind", "seconds ahead", "position",
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
       "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
        "ahead", "behind", "interval", "lapping", "traffic", "blue flag", "let him past",
        "catching", "pulling away", "closing", "older tyres",
        "fresher tyres", "tyre age", "lap old", "in the points", "out of the points",
        "seconds quicker", "seconds slower", "time loss", "time gain", "time delta"
    ],
    "team_order": [
        "let him past", "let him through", "move over",
        "hold position", "don't fight", "do not fight",
        "swap", "we will swap", "team orders", "yield",
        "stay behind", "give way", "multi-21", "hold him up", "interfere", "he has to move"
    ],
    "mechanical_issue": [
        "brake", "brakes", "brake failure", "brake cooling",
        "engine", "hydraulic", "gearbox", "drs", "drs failure",
        "puncture", "damage", "vibration", "warning", "overheating",
        "temperature", "water temp", "oil temp", "tyre pressure",
        "pressure", "loss of power", "power loss", "failure",
        "issue", "problem", "retire", "retiring", "stopped",
        "broken", "leaking", "smoke", "gps", "lost gps", 
        "difficult to drive", "undriveable"
    ],
    "acknowledgement": [
        "copy", "understood", "roger", "ok", "okay", "affirm",
        "confirmed", "noted", "yes", "no problem", "got it",
        "will do", "on it", "acknowledge"
    ],
    "other": []
}

HIGH_STRESS = ["gone completely", "completely gone", "no grip at all",
                "lost the rear completely", "nothing on the tyres",
                "tyres are dead", "tyres are gone", "fronts are gone",
                "rears are gone", "front is gone", "rear is gone",
                "falling off a cliff", "massive deg", "no tyre left",
                "not going to last", "danger zone", "falling apart",
                "flat tyre", "tire is flat", "tyre is flat",
                "damaged", "no grip on the", "definitely not cleaning up",
                "starts to degrade", "huge rear"]

MEDIUM_STRESS = ["struggling", "difficult", "dropping off", "getting worse",
                "not great", "overheating", "hot", "graining", "blistering",
                "degrading", "wearing", "going off", "losing it",
                "warm side", "getting a bit hot", "keep an eye on",
                "losing performance", "graining a little",
                "small signs of", "a bit hot", "hotter end"]

def classify_radio_message(message):
    if pd.isna(message) or message == "":
        return "other"
    message = message.lower()
    for category, keywords in CATEGORIES_DICT.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', message):
                return category
    return "other"

df["category"] = df["transcription"].apply(classify_radio_message)
print(df["category"].value_counts())

def classify_stress_level(message):
    if len(message) > 500:
        return "informational"
    if pd.isna(message) or message == "":
        return "informational"
    message = message.lower()
    for phrase in HIGH_STRESS:
        if re.search(r'\b' + re.escape(phrase) + r'\b', message):
            return "high"
    for phrase in MEDIUM_STRESS:
        if re.search(r'\b' + re.escape(phrase) + r'\b', message):
            return "medium"
    return "informational"

tyre_df = df[df["category"] == "tyre_feedback"].copy()
tyre_df["stress_level"] = tyre_df["transcription"].apply(classify_stress_level)
print(tyre_df["stress_level"].value_counts())

df["stress_level"] = None
df.loc[tyre_df.index, "stress_level"] = tyre_df["stress_level"]

# print(tyre_df[tyre_df["stress_level"] == "high"]["transcription"].sample(5).to_list())
# print(tyre_df[tyre_df["stress_level"] == "informational"]["transcription"].sample(5).to_list())

df.to_csv("data\\external\\team-radios\\classified_radios.csv", index=False)

# negation_pattern = r"\b(don't|do not|not|never|no)\b.{0,20}\b(box|push|stay out|swap)\b"
# mask = df["transcription"].str.lower().str.contains(negation_pattern, regex=True, na=False)
# print(df[mask]["transcription"].head(10).to_list())
# print(f"Total: {mask.sum()}")