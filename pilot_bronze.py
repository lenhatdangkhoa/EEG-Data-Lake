
from pyspark.sql import SparkSession, functions as F, types as T
import pandas as pd
import numpy as np
import io
import boto3
import os

AWS_REGION = "us-east-1"
BUCKET = "eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze_pilot"     
DELTA_PATH = f"s3a://{BUCKET}/bronze_delta_pilot"  # output Delta table
FS = 128.0 
s3 = boto3.client("s3")

spark = (
    SparkSession.builder
    .appName("EEG-Pilot-Bronze->Delta")
    .master("local[*]")
    # --- S3A + Delta jars (align versions with your Spark/Hadoop) ---
    .config("spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-aws:3.4.1",
                "com.amazonaws:aws-java-sdk-bundle:1.12.772",
                "io.delta:delta-spark_2.12:3.2.0"     # if on Spark 3.5.x; adjust if needed
            ]))
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    # --- S3A settings (note: timeouts are millis, not '60s') ---
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider,"
            "com.amazonaws.auth.profile.ProfileCredentialsProvider,"
            "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
    .config("spark.hadoop.fs.s3a.attempts.maximum", "10")
    .config("spark.hadoop.fs.s3a.connection.maximum", "200")
    .getOrCreate()
)

# 1) Generate a few EEG-like CSV files on S3
# -----------------------------
def make_signal_df(seconds:float, seed:int):
    N = int(seconds * FS)
    df = (spark.range(N)
          .withColumnRenamed("id", "idx")
          # pseudo EEG channels: sin/cos mixtures + small noise
          .withColumn("AF3",  F.sin(2*F.lit(3.14)*F.col("idx")/F.lit(FS)*8.0) + 0.1*F.randn(seed))
          .withColumn("AF4",  F.cos(2*F.lit(3.14)*F.col("idx")/F.lit(FS)*10.0) + 0.1*F.randn(seed+1))
          .withColumn("T7",   F.sin(2*F.lit(3.14)*F.col("idx")/F.lit(FS)*6.0) + 0.1*F.randn(seed+2))
          .withColumn("T8",   F.cos(2*F.lit(3.14)*F.col("idx")/F.lit(FS)*12.0) + 0.1*F.randn(seed+3))
          .withColumn("Pz",   F.sin(2*F.lit(3.14)*F.col("idx")/F.lit(FS)*4.0) + 0.1*F.randn(seed+4))
         )
    return df.select("AF3","AF4","T7","T8","Pz")

def upload_df_as_csv(df: pd.DataFrame, bucket: str, key: str):
    # serialize to CSV in memory (UTF-8 text)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    body_bytes = buf.getvalue().encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body_bytes, ContentType="text/csv")
    print(f"Uploaded s3://{bucket}/{key}  ({len(body_bytes)/1024:.1f} KB)")


for i in range(3):
    df = make_signal_df(seconds=3.0, seed=100+i)
    key = f"{BRONZE_PREFIX}/trial_{i}.csv"
    upload_df_as_csv(df, BUCKET, key)