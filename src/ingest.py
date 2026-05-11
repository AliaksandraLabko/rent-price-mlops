import os
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "arboreal-totem-495915-v1")
RAW_BUCKET = os.getenv("RAW_BUCKET", "arboreal-totem-495915-v1-rent-mlops-raw-data")
DATA_FILE = os.getenv("DATA_FILE", "apartments_rent_pl_2024_04-2024_06.csv")

def check_data_exists():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RAW_BUCKET)
    blob = bucket.blob(DATA_FILE)

    path = f"gs://{RAW_BUCKET}/{DATA_FILE}"

    if blob.exists():
        print(f"SUCCESS: Dataset found at {path}")
    else:
        raise FileNotFoundError(f"Dataset not found at {path}")

if __name__ == "__main__":
    check_data_exists()
