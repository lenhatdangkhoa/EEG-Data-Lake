from pyspark.sql import functions as F, types as T, Window as W, SparkSession

BRONZE_DELTA = "s3a://eeg-data-lake-khoa/bronze_delta/delta"
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

bronze = spark.read.format("delta").load(BRONZE_DELTA)

bronze = bronze.withColumn(
    "trial_id",
    F.xxhash64("synset", "image_id", "take", "session")  # 64-bit int, stable
)
w_tc = W.partitionBy("trial_id", "channel")

prep_df = (
    bronze
    .withColumn("mean_val", F.mean("value").over(w_tc))
    .withColumn("std_val",  F.stddev_samp("value").over(w_tc))
    # avoid NaN if std_val = 0
    .withColumn(
        "z",
        F.when(F.col("std_val") == 0, F.lit(0.0))
         .otherwise((F.col("value") - F.col("mean_val")) / F.col("std_val"))
    )
    .withColumn(
        "is_outlier",
        F.abs(F.col("z")) > F.lit(6.0)   # same threshold as pilot
    )
)

silver_clean = (
    prep_df
    .filter(~F.col("is_outlier"))       # drop outliers; or keep them if you just want flags
    .select(
        "trial_id",
        "headset", "synset", "image_id", "take", "session",
        "channel",
        "sample_idx", "time_sec",
        "value", "z", "is_outlier",
        "source_file"
    )
)
(
    silver_clean
    # optional: adjust parallelism if needed
    # .repartition("synset")   # or "trial_id" if you want
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("synset", "channel")   # coarser than bronze; good balance
    .save(SILVER_DELTA)
)

print(f"Silver Delta written to: {SILVER_DELTA}")