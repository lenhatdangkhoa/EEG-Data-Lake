from pyspark.sql import functions as F, types as T, Window as W, SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
gold = spark.read.format("delta").load(GOLD_DELTA)

label_col = "synset"

numeric_features = [
    "n_samples",
    "duration_sec",
    "duration_from_count_sec",
    "mean_value",
    "std_value",
    "mean_abs_value",
    "min_value",
    "max_value",
    "median_value",
    "p95_value",
    "mean_z",
    "std_z",
    "min_z",
    "max_z",
    "samples_per_sec",
]

gold_clean = gold.filter(F.col("duration_sec") > 0)

# 1. Label indexer
label_indexer = StringIndexer(
    inputCol=label_col,
    outputCol="label",
    handleInvalid="skip"   # skip unseen / null labels
).fit(gold_clean)

print("Finished label indexing")
# 2. Assemble features
assembler = VectorAssembler(
    inputCols=numeric_features,
    outputCol="features_raw",
    handleInvalid="keep"
)

print("Finished assembling features")
# 3. Scale features
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

print("Finished scaling features")
# 4. Classifier
lr = LogisticRegression(
    labelCol="label",
    featuresCol="features",
    maxIter=50,
    regParam=0.01,
    elasticNetParam=0.0  # L2
)
pipeline = Pipeline(stages=[label_indexer, assembler, scaler, lr])

train_df, test_df = gold_clean.randomSplit([0.8, 0.2], seed=42)
print(f"Training on {train_df.count()} rows, testing on {test_df.count()} rows")
model = pipeline.fit(train_df)
print("Finished model training")
predictions = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test accuracy (channel-level): {accuracy:.4f}")

for m in ["f1", "weightedPrecision", "weightedRecall"]:
    ev = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=m
    )
    print(m, ev.evaluate(predictions))

MODEL_PATH = "s3a://eeg-data-lake-khoa/models/rf_synset_trial_channel"

model.write().overwrite().save(MODEL_PATH)
print(f"Saved RF model to {MODEL_PATH}")


PREDICTIONS_DELTA = "s3a://eeg-data-lake-khoa/gold_predictions_trial_channel"

(predictions
 .select(
     "trial_id",
     "channel",
     "synset",
     "image_id",
     "prediction",
     "probability"
 )
 .write
 .format("delta")
 .mode("overwrite")
 .save(PREDICTIONS_DELTA)
)
