import boto3
import os
import mne 
import numpy as np
import scipy
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf

# AWS Configuration
AWS_REGION = "us-east-1"
BUCKET="eeg-data-lake-khoa"
BRONZE_PREFIX = "s3a://eeg-data-lake-khoa/bronze/*.csv"
SILVER_PREFIX = "s3a://eeg-data-lake-khoa/silver/"
FS = 128  # sampling freq

s3 = boto3.client('s3', region_name=AWS_REGION)
# Create a SparkSession
spark = SparkSession.builder \
    .appName("BronzeToSilver") \
    .master("local[*]") \
    .config("spark.jars.packages","org.apache.hadoop:hadoop-aws:3.4.1") \
    .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint","s3.amazonaws.com") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
        "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider") \
    .getOrCreate()

bronze_df = (spark.read
             .option("header", "false")
             .option("inferSchema", "false")      # speed & schema stability
             .csv(BRONZE_PREFIX)
             .withColumn("file", F.input_file_name()))


# cast numeric sample columns to double
sample_cols = [c for c in bronze_df.columns if c not in ("file", "_c0")]
for c in sample_cols:
    bronze_df = bronze_df.withColumn(c, F.col(c).cast("double"))

stack_expr = "stack({n}, {pairs}) as (sample_idx, value)".format(
    n=len(sample_cols),
    pairs=", ".join([f"'{i}', `{c}`" for i, c in enumerate(sample_cols)])
)
long_df = (
    bronze_df
    .select(
        F.col("file"),
        F.col("_c0").alias("channel"),
        F.expr(stack_expr)
    )
    .withColumn("timepoint", F.col("sample_idx").cast("int"))
    .drop("sample_idx")
    .filter(F.col("value").isNotNull())
)

meta_df = (
    long_df
    .withColumn("synset",  F.regexp_extract("file", r"Insight_(n\d{8})_", 1))
    .withColumn("image_id",F.regexp_extract("file", r"Insight_n\d{8}_(\d+)_", 1))
    .withColumn("repeat",  F.regexp_extract("file", r"Insight_n\d{8}_\d+_(\d+)_", 1))
    .withColumn("session", F.regexp_extract("file", r"Insight_n\d{8}_\d+_\d+_(\d+)\.csv$", 1))
)

B, A = butter(4, [1/(FS/2), 40/(FS/2)], btype="band")

schema = T.StructType([
    T.StructField("file", T.StringType()),
    T.StructField("synset", T.StringType()),
    T.StructField("image_id", T.StringType()),
    T.StructField("repeat", T.StringType()),
    T.StructField("session", T.StringType()),
    T.StructField("timepoint", T.IntegerType()),
    T.StructField("time_sec", T.DoubleType()),
    T.StructField("channel", T.StringType()),
    T.StructField("value_z", T.DoubleType()),
])

def process_file(pdf: pd.DataFrame) -> pd.DataFrame:
    # pdf columns: file, channel, value, timepoint, synset, image_id, repeat, session
    pdf = pdf.dropna(subset=["value"]).sort_values(["timepoint", "channel"])
    pivot = pdf.pivot(index="timepoint", columns="channel", values="value").sort_index()
    X = pivot.to_numpy(dtype=float, copy=False)

    # If too short for filtfilt, fall back to raw
    try:
        Xf = filtfilt(B, A, X, axis=0)
    except Exception:
        Xf = X

    mu = Xf.mean(axis=0, keepdims=True)
    sd = Xf.std(axis=0, ddof=0, keepdims=True)
    Xz = (Xf - mu) / (sd + 1e-8)

    out = pd.DataFrame(Xz, columns=pivot.columns, index=pivot.index).reset_index()
    out["time_sec"] = out["timepoint"] / FS
    out = out.melt(id_vars=["timepoint", "time_sec"], var_name="channel", value_name="value_z")

    # attach metadata
    f = pdf["file"].iloc[0]
    syn = pdf["synset"].iloc[0]
    img = pdf["image_id"].iloc[0]
    rep = pdf["repeat"].iloc[0]
    ses = pdf["session"].iloc[0]

    out.insert(0, "file", f)
    out.insert(1, "synset", syn)
    out.insert(2, "image_id", img)
    out.insert(3, "repeat", rep)
    out.insert(4, "session", ses)

    return out[["file","synset","image_id","repeat","session",
                "timepoint","time_sec","channel","value_z"]]

silver_long = (
    meta_df
    .groupBy("file")                # per trial
    .applyInPandas(process_file, schema=schema)
)

# ==== Write Silver Parquet ====
(
    silver_long
    .repartition("synset", "session")           # cluster by partition keys
    .write
    .mode("append")                              
    .option("compression", "snappy")
    .option("maxRecordsPerFile", 500_000)       
    .partitionBy("synset", "session")            # directory layout
    .parquet(SILVER_PREFIX)
)

spark.stop()
