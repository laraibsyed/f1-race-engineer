"""Produces driver_taxonomy_master_final.csv. The aggression/defense file lives in the bucket
(uploaded earlier under processed/), the other two are local - they were never uploaded."""
import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
local_vm_path = "driver_taxonomy_sampled_scores_bucket.csv"
bucket.blob("processed/driver_taxonomy_sampled_scores.csv").download_to_filename(local_vm_path)
vm = pd.read_csv(local_vm_path)[["driver", "aggression_level", "defensive_strength"]]

simple = pd.read_csv("driver_simple_metrics_v2.csv")[["driver", "tyre_management",
                                                        "consistency_factor", "wet_weather_skill"]]
pressure = pd.read_csv("driver_pressure_risk_tolerance.csv")[["driver", "pressure_risk_tolerance"]]

master = vm.merge(simple, on="driver", how="outer").merge(pressure, on="driver", how="outer")
master.to_csv("driver_taxonomy_master_final.csv", index=False)
print(f"[save] driver_taxonomy_master_final.csv - {len(master)} drivers")