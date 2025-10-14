from pyspark.sql import functions as F, types as T, Window as W, SparkSession

BUCKET = "eeg-data-lake-khoa"
FS = 128.0                              # sampling rate (Hz)
WIN_SEC = 1.0                           # analysis window length (s)
OVERLAP = 0.5                           # 50% overlap
STEP_SEC = WIN_SEC * (1.0 - OVERLAP)    # hop length (s)
SILVER_DELTA = f"s3a://{BUCKET}/silver_delta_pilot"
GOLD_FEATURES_DELTA = f"s3a://{BUCKET}/gold_delta_pilot"
GOLD_SIGNAL_DS_DELTA = f"s3a://{BUCKET}/gold_signal_ds_delta"

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

silver = spark.read.format("delta").load(SILVER_DELTA).cache()

with_epochs = (
    silver
    .withColumn("epoch_id", F.floor(F.col("time_sec") / F.lit(STEP_SEC)).cast("long"))
    .withColumn("win_start_sec", (F.col("epoch_id") * F.lit(STEP_SEC)).cast("double"))
    .withColumn("win_end_sec",   (F.col("epoch_id") * F.lit(STEP_SEC) + F.lit(WIN_SEC)).cast("double"))
    # keep only samples that actually fall inside the epoch's 1s window
    .where((F.col("time_sec") >= F.col("win_start_sec")) & (F.col("time_sec") < F.col("win_end_sec")))
)

# Row-wise helpers for features
signed = with_epochs.withColumn("sign", F.when(F.col("value") >= 0, 1).otherwise(-1))
w_seq = W.partitionBy("trial_id", "channel", "epoch_id").orderBy(F.col("sample_idx"))
with_lag = (
    signed
    .withColumn("sign_prev", F.lag("sign").over(w_seq))
    .withColumn("is_change", F.when(F.col("sign_prev").isNotNull() & (F.col("sign") != F.col("sign_prev")), 1).otherwise(0))
)
# Basic stats
agg = (
    with_lag
    .groupBy("trial_id", "channel", "epoch_id", "win_start_sec", "win_end_sec")
    .agg(
        F.count("*").alias("n"),
        F.sum(F.when(F.col("value").isNull(), 1).otherwise(0)).alias("n_nan"),
        F.avg("value").alias("mean"),
        F.stddev_samp("value").alias("std"),
        F.sqrt(F.avg(F.col("value")*F.col("value"))).alias("rms"),
        (F.max("value") - F.min("value")).alias("p2p"),
        F.skewness("value").alias("skew"),
        F.kurtosis("value").alias("kurtosis"),
        F.sum("is_change").alias("zcr_count")
    )
    .withColumn("nan_ratio", F.col("n_nan") / F.col("n"))
    .withColumn("zcr", F.col("zcr_count") / F.lit(WIN_SEC))  # changes per second
)
with_dx = (
    with_lag
    .withColumn("x", F.col("value"))
    .withColumn("x_prev", F.lag("value").over(w_seq))
    .withColumn("dx", F.when(F.col("x_prev").isNull(), F.lit(0.0)).otherwise(F.col("x") - F.col("x_prev")))
    .withColumn("dx_prev", F.lag("dx").over(w_seq))
    .withColumn("d2x", F.when(F.col("dx_prev").isNull(), F.lit(0.0)).otherwise(F.col("dx") - F.col("dx_prev")))
)

hj_agg = (
    with_dx
    .groupBy("trial_id", "channel", "epoch_id")
    .agg(
        F.variance("x").alias("var_x"),
        F.variance("dx").alias("var_dx"),
        F.variance("d2x").alias("var_d2x")
    )
    .withColumn("hj_activity", F.col("var_x"))
    .withColumn("hj_mobility", F.sqrt(F.when(F.col("var_x") == 0, 0.0).otherwise(F.col("var_dx")/F.col("var_x"))))
    .withColumn(
        "hj_complexity",
        F.when(F.col("var_dx") == 0, F.lit(0.0))
         .otherwise( F.sqrt(F.col("var_d2x")/F.col("var_dx")) / F.when(F.col("hj_mobility")==0, F.lit(1.0)).otherwise(F.col("hj_mobility")) )
    )
)
features = (
    agg.join(hj_agg, on=["trial_id","channel","epoch_id"], how="left")
)
features = (
    features
    .withColumn("flatline_flag",
        F.when((F.col("p2p") <= F.lit(1e-6)) | (F.col("std").isNull()) | (F.col("std") <= F.lit(1e-12)), F.lit(True)).otherwise(F.lit(False))
    )
    .withColumn("highvar_flag",
        F.when(F.col("std") > F.lit(100.0), F.lit(True)).otherwise(F.lit(False))  # tune threshold for your scale
    )
)
(
    features
    .repartition("channel")  # good parallelism & partitioning
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("channel")
    .save(GOLD_FEATURES_DELTA)
)

print(f"[gold] Features written to: {GOLD_FEATURES_DELTA}")
spark.read.format("delta").load(GOLD_FEATURES_DELTA).createOrReplaceTempView("gold_features")