import pandas as pd

COMPOUND_MAPPING = {
    "HYPERSOFT": "SOFT",
    "SUPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",

}

def clean_compound(series: pd.Series) -> pd.Series:
    return series.replace(COMPOUND_MAPPING)

if __name__ == "__main__":

    test = pd.Series(["HYPERSOFT", "SOFT", "SUPERSOFT", "MEDIUM", "ULTRASOFT", "HARD", None])
    print("Before:", test.tolist())
    print("After: ", clean_compound(test).tolist())
