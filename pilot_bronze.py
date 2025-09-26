
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
                "io.delta:delta-spark_2.13:4.0.0"     
            ]))
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.port", "52345")         # any free high port
    .config("spark.blockManager.port", "52346")   # next free port
    .config("spark.ui.port", "4045")         
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
def make_signal_df(seconds: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    return pd.DataFrame({
        "AF3":  np.sin(2*np.pi*t*8.0)  + 0.1*rng.standard_normal(n),
        "AF4":  np.cos(2*np.pi*t*10.0) + 0.1*rng.standard_normal(n),
        "T7":   np.sin(2*np.pi*t*6.0)  + 0.1*rng.standard_normal(n),
        "T8":   np.cos(2*np.pi*t*12.0) + 0.1*rng.standard_normal(n),
        "Pz":   np.sin(2*np.pi*t*4.0)  + 0.1*rng.standard_normal(n),
    })


def upload_df_as_csv(df: pd.DataFrame, bucket: str, key: str):
    # serialize to CSV in memory (UTF-8 text)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    body_bytes = buf.getvalue().encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body_bytes, ContentType="text/csv")
    print(f"Uploaded s3://{bucket}/{key}  ({len(body_bytes)/1024:.1f} KB)")


# for i in range(3):
#     df = make_signal_df(seconds=3.0, seed=100+i)
#     key = f"{BRONZE_PREFIX}/trial_{i}.csv"
#     upload_df_as_csv(df, BUCKET, key)
 
# ---- 1) Read ALL CSVs in that directory (headered, same schema) ----
src_path = f"s3a://{BUCKET}/bronze_pilot/*.csv"   # flat dir; wildcard OK

schema = T.StructType([
    T.StructField("AF3",  T.DoubleType(), True),
    T.StructField("AF4",  T.DoubleType(), True),
    T.StructField("T7",   T.DoubleType(), True),
    T.StructField("T8",   T.DoubleType(), True),
    T.StructField("Pz",   T.DoubleType(), True),

])
df = (spark.read
      .option("header", "true")
      .schema(schema)              # use .option("inferSchema","true") if schema varies
      .csv(src_path)
      .withColumn("source_file", F.input_file_name())
)
df.printSchema()
df.show(20, truncate=False)  # first 20 rows