import pandas as pd

# 2018-only legacy Pirelli tyre naming -> current standard naming.
# Mapping confirmed against real data: 2018 used ONLY the left-hand side
# names, 2019 onward used ONLY the right-hand side names -- no overlap,
# so this mapping is safe to apply unconditionally by name (not by year).
COMPOUND_MAPPING = {
    "HYPERSOFT": "SOFT",
    "SUPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",
    # Standard names pass through unchanged (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
}


def clean_compound(series: pd.Series) -> pd.Series:
    return series.replace(COMPOUND_MAPPING)


if __name__ == "__main__":
    # quick smoke test
    test = pd.Series(["HYPERSOFT", "SOFT", "SUPERSOFT", "MEDIUM", "ULTRASOFT", "HARD", None])
    print("Before:", test.tolist())
    print("After: ", clean_compound(test).tolist())