from pyspark.sql import functions as F, types as T, Window as W, SparkSession

GOLD_DELTA = "s3a://eeg-data-lake-khoa/gold_delta"
SILVER_DELTA = "s3a://eeg-data-lake-khoa/silver_delta_mindbigdata"

FS = 128.0  # Hz
spark = (
    SparkSession.builder
    .appName("EEG-Pilot-Bronze->Delta")
    .master("local[*]")
    .config("spark.driver.memory", "8g")        
    .config("spark.executor.memory", "8g")
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
silver = spark.read.format("delta").load(SILVER_DELTA)

gold_trial_channel = (
    silver
    .groupBy(
        "trial_id",
        "channel",
        "synset",
        "image_id",
        "take",
        "session",
        "headset"
    )
    .agg(
        # basic counts / timing
        F.count("*").alias("n_samples"),
        F.min("time_sec").alias("t_start_sec"),
        F.max("time_sec").alias("t_end_sec"),
        
        # duration from time axis and from sample count
        (F.max("time_sec") - F.min("time_sec")).alias("duration_sec"),
        (F.count("*") / F.lit(FS)).alias("duration_from_count_sec"),
        
        # value statistics
        F.mean("value").alias("mean_value"),
        F.stddev_samp("value").alias("std_value"),
        F.mean(F.abs("value")).alias("mean_abs_value"),
        F.min("value").alias("min_value"),
        F.max("value").alias("max_value"),
        F.expr("percentile_approx(value, 0.5)").alias("median_value"),
        F.expr("percentile_approx(value, 0.95)").alias("p95_value"),
        
        # z-score stats (after your per-trial z-normalization)
        F.mean("z").alias("mean_z"),
        F.stddev_samp("z").alias("std_z"),
        F.min("z").alias("min_z"),
        F.max("z").alias("max_z"),
    )
    # helpful QC ratios
    .withColumn(
        "samples_per_sec",
        F.col("n_samples") / F.col("duration_sec")
    )
)

(
    gold_trial_channel
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("synset", "channel")
    .save(GOLD_DELTA)
)

print(f"Gold Delta (trial x channel features) written to: {GOLD_DELTA}")
