from pyspark.sql import functions as F, types as T, Window as W, SparkSession

BUCKET = "eeg-data-lake-khoa"
GOLD_FEATURES_DELTA = f"s3a://{BUCKET}/gold_delta_pilot"

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

spark.read.format("delta").load(GOLD_FEATURES_DELTA).createOrReplaceTempView("gold_features")
# 1) Count epochs by trial/channel
spark.sql("""
SELECT trial_id, channel, COUNT(*) AS n_epochs,
       SUM(CASE WHEN flatline_flag THEN 1 ELSE 0 END) AS n_flat,
       SUM(CASE WHEN highvar_flag THEN 1 ELSE 0 END)  AS n_highvar
FROM gold_features
GROUP BY trial_id, channel
ORDER BY trial_id, channel
""").show(50, truncate=False)

# 2) Look at a few feature rows
spark.sql("""
SELECT trial_id, channel, epoch_id, win_start_sec, mean, std, rms, p2p, zcr,
       hj_activity, hj_mobility, hj_complexity, nan_ratio, flatline_flag, highvar_flag
FROM gold_features
ORDER BY trial_id, channel, epoch_id
LIMIT 20
""").show(truncate=False)

features = spark.read.format("delta").load(f"{GOLD_FEATURES_DELTA}")
features.show(10, truncate=False)