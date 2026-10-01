
import io
import os

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

SAMPLE_SESSIONS = [
    ("2018", "Australian_Grand_Prix", "R"),
    ("2021", "Azerbaijan_Grand_Prix", "R"),
    ("2022", "Australian_Grand_Prix", "R"),
    ("2023", "Qatar_Grand_Prix", "S"),
    ("2024", "Abu_Dhabi_Grand_Prix", "R"),
]

SC_VSC_KEYWORDS = ["SAFETY CAR", "VIRTUAL SAFETY CAR", "VSC"]

def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None

def find_file(year: str, race: str, session: str, filename: str) -> str | None:
    prefix = f"raw/fastf1/{year}/{race}/{session}/"
    for b in client.list_blobs(BUCKET_NAME, prefix=prefix):
        if b.name.endswith(filename):
            return b.name
    return None

def inspect_session(year: str, race: str, session: str):
    print("=" * 70)
    print(f"{year}/{race}/{session}")
    print("=" * 70)

    path = find_file(year, race, session, "messages.csv")
    if not path:
        print(f"  NOT FOUND under raw/fastf1/{year}/{race}/{session}/")
        return

    print(f"  Found: {path}")
    df = read_csv_robust(bucket.blob(path).download_as_bytes())
    if df is None or df.empty:
        print("  Empty or unreadable.")
        return

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print()

    for col in df.columns:
        series = df[col]
        pct_null = round(series.isna().mean() * 100, 1)
        non_null = series.dropna()
        if len(non_null) and not pd.api.types.is_numeric_dtype(series):
            uniques = non_null.unique()
            preview = sorted(map(str, uniques))[:8] if len(uniques) <= 30 else f"{len(uniques)} unique values"
        elif len(non_null):
            preview = f"min={non_null.min()}, max={non_null.max()}"
        else:
            preview = "N/A"
        print(f"    {col:20s} null={pct_null:5.1f}%  {preview}")

    text_col = next((c for c in df.columns if c.lower() == "message"), None)
    if text_col is None:
        print("\n  No obvious 'Message' text column found -- inspect columns above manually.")
        return

    print(f"\n  SC/VSC-related rows (matching on '{text_col}' column):")
    mask = df[text_col].astype(str).str.upper().apply(lambda s: any(k in s for k in SC_VSC_KEYWORDS))
    sc_vsc_rows = df[mask]
    if sc_vsc_rows.empty:
        print("    None found in this session.")
    else:
        print(sc_vsc_rows.to_string())

    print()

if __name__ == "__main__":
    for year, race, session in SAMPLE_SESSIONS:
        inspect_session(year, race, session)
