from pyspark.sql import functions as F, types as T, Window as W, SparkSession

FS = 128.0
SRC_GLOB = "s3a://eeg-data-lake-khoa/bronze/"  
DEST = "s3a://eeg-data-lake-khoa/bronze_delta"
OUT_DELTA = f"{DEST}/delta"
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

raw = (spark.read
      .option("header","false")
      .option("recursiveFileLookup","true")
      .csv(SRC_GLOB)
      .withColumn("source_file", F.input_file_name()))
data_cols = [c for c in raw.columns if c != "source_file"]

rx = r".*MindBigData_Imagenet_([^_/]+)_(n\d{8})_(\d+)_(\d+)_(\d+)\.csv"
df = (
    raw
    .withColumn("headset",  F.regexp_extract("source_file", rx, 1))
    .withColumn("synset",   F.regexp_extract("source_file", rx, 2))
    .withColumn("image_id", F.regexp_extract("source_file", rx, 3).cast("int"))
    .withColumn("take",     F.regexp_extract("source_file", rx, 4).cast("int"))
    .withColumn("session",  F.regexp_extract("source_file", rx, 5).cast("int"))
)
row_arr = F.array(*[F.col(c).cast("string") for c in data_cols])
# 3) Split each line: first token = channel label; rest are sample values
df = (
    df.withColumn("row_arr", row_arr)
      .withColumn("channel", F.col("row_arr").getItem(0))
      .withColumn("samples_str", F.slice(F.col("row_arr"), 2, 100000))  # drop channel
      # convert to array<double>
      .withColumn("samples",
                  F.expr("transform(samples_str, x -> cast(x as double))"))
      .drop("row_arr", "samples_str")
)

# 4) Explode to one row per sample
long_df = (
    df.select(
        "source_file",
        "headset", "synset", "image_id", "take", "session", "channel",
        F.posexplode("samples").alias("sample_idx", "value")  # 2 aliases here
    )
    .withColumn("time_sec", F.col("sample_idx") / F.lit(FS))
)
(
    long_df
    .repartition("synset","image_id","channel")
    .write.format("delta")
    .mode("overwrite")
    .partitionBy("synset","image_id","channel")
    .save(OUT_DELTA)
)
print(f"Delta written to {OUT_DELTA}")