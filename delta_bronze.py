import re
from pyspark.sql import SparkSession, functions as F, types as T
import pyspark
from delta.tables import DeltaTable

AWS_REGION = "us-east-1"
BUCKET = "eeg-data-lake-khoa"

RAW = f"s3a://{BUCKET}/bronze/*.csv"              # MindBigData_Imagenet_Insight_*.csv
BRONZE_DELTA = f"s3a://{BUCKET}/bronze_delta"          # destination Delta table

FS = 128.0  # Hz
EXPECTED_CHANNELS = ["AF3","AF4","T7","T8","Pz"]      


# MindBigData_Imagenet_Insight_n09835506_15262_1_20.csv
FNAME_RE = r".*MindBigData_Imagenet_Insight_(n\d+)_(\d+)_(\d+)_(\d+)\.csv$"
# ==========================


spark = (
    SparkSession.builder
    .appName("BronzeToSilver") 
    .master("local[*]") 
    .config("spark.jars.packages","org.apache.hadoop:hadoop-aws:3.4.1") 
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem") 
    .config("spark.hadoop.fs.s3a.endpoint","s3.amazonaws.com")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider")
    .getOrCreate()
)
# 1) Read raw files as TEXT 
raw = (spark.read.text(RAW)
       .withColumn("_source_file", F.input_file_name())
       .withColumn("_ingest_ts", F.current_timestamp())
       .withColumn("_ingest_date", F.to_date("_ingest_ts")))

# 2) Extract labels from filename
synset_id = F.regexp_extract(F.col("_source_file"), FNAME_RE, 1)
image_id = F.regexp_extract(F.col("_source_file"), FNAME_RE, 2).cast("int")
take_idx = F.regexp_extract(F.col("_source_file"), FNAME_RE, 3).cast("int")
global_session = F.regexp_extract(F.col("_source_file"), FNAME_RE, 4).cast("int")

lines = raw.select(
    F.col("value").alias("line"),
    "_source_file","_ingest_ts","_ingest_date",
    synset_id.alias("synset_id"),
    image_id.alias("image_id"),
    take_idx.alias("take_idx"),
    global_session.alias("global_session"),
)

# 3) Split each text line: first token = channel, the rest = samples
arr = F.split(F.col("line"), ",")
lines2 = (lines
    .withColumn("channel", arr.getItem(0))
    .withColumn("values", F.expr("slice(split(line, ','), 2, size(split(line, ',')) - 1)"))
    .drop("line"))

# 4) one row per sample; derive sample_idx and time_sec
samples = (
    lines2
    .selectExpr(
        "synset_id","image_id","take_idx","global_session",
        "channel","_source_file","_ingest_ts","_ingest_date",
        "posexplode(values) as (sample_idx, value_str)"         # ✅ SQL form
    )
    .withColumn("value", F.col("value_str").cast("double"))
    .drop("value_str")
    .withColumn("time_sec", F.col("sample_idx").cast("double")/F.lit(128.0))
)

# 5) Clean (known channels only; drop NAs)
bronze_df = (samples
    .where(F.col("channel").isin(*EXPECTED_CHANNELS))
    .dropna(subset=["value"])
)
(bronze_df
  .repartition("synset_id","_ingest_date")
  .write.format("delta").mode("append")
  .partitionBy("synset_id","_ingest_date")   # only on the very first write
  .save(BRONZE_DELTA))


print("Bronze ingest complete.")
