import pyarrow.parquet
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv
import io

BUCKET_NAME = "f1-race-engineer-bucket"

load_dotenv()

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

first_batch = True


for i in range(5):
    file_name = f"raw/team-radios/train-{i:05d}-of-00005.parquet"
    radios_blob = bucket.blob(file_name)

    data_bytes = radios_blob.download_as_bytes()

    parquet_file = pyarrow.parquet.ParquetFile(io.BytesIO(data_bytes))
    batches = parquet_file.iter_batches(batch_size=1000, columns=['id', 'driver_id', 'racing_number', 'grand_prix', 'race_id',
       'session_date', 'message_timestamp', 'transcription'])

    for batch in batches:
        print(f"Downloading shard {i}...")
        df_chunk = batch.to_pandas()
        df = df_chunk.to_csv("data\\external\\team-radios\\transcriptions.csv", mode="a", header=first_batch, index=False)
        first_batch = False

    print(f"Done {i}!")

print("All done!")