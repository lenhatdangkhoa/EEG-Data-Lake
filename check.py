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
df = (spark.read
      .option("header", "false")        # set False if no header row
      .option("inferSchema", "true")   # for a quick peek; define a schema later for production
      .csv(RAW))

df.printSchema()
df.show(20, truncate=False)  # first 20 rows