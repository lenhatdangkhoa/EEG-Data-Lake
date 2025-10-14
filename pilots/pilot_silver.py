# ---------- Silver: combine + preprocess + write to S3 as Delta ----------
from pyspark.sql import functions as F, types as T, Window as W, SparkSession

BUCKET = "eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze_pilot"
# --- paths ---
SRC_GLOB      = f"s3a://{BUCKET}/{BRONZE_PREFIX}/*.csv"   # all bronze CSVs
SILVER_DELTA  = f"s3a://{BUCKET}/silver_delta_pilot"      # Delta Silver (recommended)
SILVER_EXPORT = f"s3a://{BUCKET}/silver_pilot_export"     # optional CSV/Parquet export

FS = 128.0  # sampling freq (Hz)

spark = spark = (
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

# --- schema & channels (edit if your bronze schema changes) ---
channels = ["AF3","AF4","T7","T8","Pz"]
schema = T.StructType([T.StructField(c, T.DoubleType(), True) for c in channels])

# 1) Read ALL bronze files
bronze_df = (
    spark.read
         .option("header", "true")
         .schema(schema)
         .csv(SRC_GLOB)
         .withColumn("source_file", F.input_file_name())
)

# 2) Extract trial_id from filename and make a per-file sample index (0..n-1)
trial_rx = F.regexp_extract(F.col("source_file"), r"trial_(\d+)\.csv", 1)
df = (
    bronze_df
    .withColumn("trial_id", trial_rx.cast("int"))
    .withColumn("_mid", F.monotonically_increasing_id())
)
w_idx = W.partitionBy("source_file").orderBy(F.col("_mid"))
df = (
    df.withColumn("sample_idx", F.row_number().over(w_idx) - F.lit(1))
      .drop("_mid")
      .withColumn("time_sec", F.col("sample_idx") / F.lit(FS))
)

# 3) Wide -> long (tidy) format: (trial_id, channel, sample_idx, time_sec, value)
stack_expr = "stack({n}, {pairs}) as (channel, value)".format(
    n=len(channels),
    pairs=", ".join([f"'{c}', `{c}`" for c in channels])
)
long_df = (
    df.select("trial_id", "source_file", "sample_idx", "time_sec", *channels)
      .selectExpr("trial_id", "source_file", "sample_idx", "time_sec", stack_expr)
      .withColumn("channel", F.col("channel").cast("string"))
      .withColumn("value",   F.col("value").cast("double"))
)

# 4) Preprocessing
#    - per (trial_id, channel) z-score
#    - outlier flag |z|>6 (tunable); we’ll drop those rows in the "clean" view
w_tc = W.partitionBy("trial_id", "channel")
prep_df = (
    long_df
    .withColumn("mean_val",  F.mean("value").over(w_tc))
    .withColumn("std_val",   F.stddev_samp("value").over(w_tc))
    .withColumn("z",         (F.col("value") - F.col("mean_val")) / F.col("std_val"))
    .withColumn("is_outlier", F.when(F.abs(F.col("z")) > F.lit(6.0), F.lit(True)).otherwise(F.lit(False)))
)

silver_clean = (
    prep_df
    .filter(~F.col("is_outlier"))   # keep this; or comment out if you prefer to retain and just flag
    .select(
        "trial_id", "channel", "sample_idx", "time_sec",
        "value", "z", "is_outlier", "source_file"
    )
)

# 5) Write Silver to S3 as Delta (partitioned for fast scans)
(
    silver_clean
    .repartition("trial_id","channel")               # good write parallelism
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("trial_id","channel")
    .save(SILVER_DELTA)
)

print(f"Delta Silver written to: {SILVER_DELTA}")

# Quick sanity read
spark.read.format("delta").load(SILVER_DELTA).show(10, truncate=False)

(
    silver_clean
    .write.mode("overwrite")
    .parquet(f"{SILVER_EXPORT}/parquet")
)


print(f"Parquet+CSV exports written under: {SILVER_EXPORT}")
