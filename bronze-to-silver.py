import boto3
import os
import mne 
import numpy as np
import scipy
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


# AWS Configuration
AWS_REGION = "us-east-1"
BUCKET="eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze/"
SILVER_PREFIX = "silver/"

s3 = boto3.client('s3', region_name=AWS_REGION)

def bandpass_filter(data, low=1, high=40, fs=128, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data)

def bronze_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a bronze EEG DataFrame to a silver DataFrame.
    Bronze format: first column = channel names, subsequent columns = timepoints
    Silver format: index = timepoints, columns = channel names
    """
    df_silver = df.set_index(0).T   # first col = channel names, then transpose
    df_silver.index.name = "timepoint"
    df_silver = df_silver.dropna()

    df_silver = df_silver.apply(lambda col: pd.Series(bandpass_filter(col.values),
                                        index=df_silver.index), axis=0)

    # Standardize each channel
    scaler = StandardScaler()
    df_silver[df_silver.columns] = scaler.fit_transform(df_silver)

    return df_silver

def silver_key_for(key: str) -> str:
    # mirror path, swap prefix, change extension to parquet
    base = os.path.basename(key).rsplit(".", 1)[0] + ".parquet"
    # keep any subfolders under bronze/
    tail = key[len(BRONZE_PREFIX):].rsplit(".", 1)[0] + ".parquet"
    return f"{SILVER_PREFIX}{tail}"

def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    
# ---------- helpers ----------
def list_csv_keys(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".csv"):
                return_key = key
                yield return_key

# Main processing loop
all_keys = list(list_csv_keys(BUCKET, BRONZE_PREFIX))
print("Bronze CSV file count:", len(all_keys))

processed = 0
for i, key in enumerate(all_keys, 1):
    out_key = silver_key_for(key)
    print(out_key)
    if object_exists(BUCKET, out_key):
        print(f"[{i}/{len(all_keys)}] Skip (exists): s3://{BUCKET}/{out_key}")
        continue

    src = f"s3://{BUCKET}/{key}"
    print(f"[{i}/{len(all_keys)}] Processing: {src}")

    # read bronze directly from S3 (needs s3fs)
    df_raw = pd.read_csv(src, header=None)

    # transform
    df_silver = bronze_to_silver(df_raw)

    # write parquet back to S3 (needs pyarrow + s3fs)
    dst = f"s3://{BUCKET}/{out_key}"
    df_silver.to_parquet(dst, index=False)

    processed += 1

print(f"Wrote {processed} Silver parquet files to s3://{BUCKET}/{SILVER_PREFIX}")
