from pyspark.sql import functions as F, types as T, Window as W, SparkSession
import clip
import torch, numpy as np, pandas as pd
from PIL import Image

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




gold = spark.read.format("delta").load("s3a://eeg-data-lake-khoa/gold_delta_pilot")
# gold columns (from your job): trial_id, channel, epoch_id, win_start_sec, win_end_sec, mean, std, ... + QC flags

# Choose a peri-stimulus window relative to onset (e.g., [0.1, 0.8] s)
REL_START = 0.10
REL_END   = 0.80


feature_cols = ["mean","std","rms","p2p","zcr","hj_mobility","hj_complexity"]

agg_per_chan = (gold
    .where((~F.col("flatline_flag")) & (~F.col("highvar_flag")) & (F.col("nan_ratio") <= 0.05))
    .groupBy("trial_id","channel")
    .agg(*[F.avg(c).alias(f"{c}_avg") for c in feature_cols])
)

wide = (agg_per_chan
    .groupBy("trial_id")
    .pivot("channel")
    .agg(*[F.first(f"{c}_avg") for c in feature_cols])
)

vec = wide.select("trial_id",
                  F.array(*[F.col(c).cast("double") for c in sorted([c for c in wide.columns if c != "trial_id"])])
                  .alias("eeg_vec"))


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def make_noise_image(seed, size=224):
    rng = np.random.default_rng(seed)
    # random smooth noise (blurred) or simple gradients
    img = (rng.normal(0.5, 0.2, (size, size, 3)).clip(0,1) * 255).astype(np.uint8)
    return Image.fromarray(img)

K = 3  # gallery size
gallery_ids = [f"synimg_{i:04d}" for i in range(K)]

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")  # faster toPandas
vec_pd = vec.select("trial_id", "eeg_vec").toPandas()

print("gold.count()", gold.count())
print("agg_per_chan.count()", agg_per_chan.count())
print("wide.count()", wide.count())
print("vec.count()", vec.count())

X = np.stack(vec_pd["eeg_vec"].to_numpy()).astype(np.float32)  # [N, D_eeg]
trial_ids = vec_pd["trial_id"].to_numpy()

with torch.no_grad():
    # Build a batch of K synthetic images → CLIP embeddings
    batch = torch.stack([preprocess(make_noise_image(1234 + i)) for i in range(K)]).to(device)
    G = model.encode_image(batch)              # [K, 512]
    G = G / G.norm(dim=-1, keepdim=True)       # normalize for cosine
G_mat = G.cpu().numpy().astype(np.float32)   

gid_idx = np.abs(np.vectorize(hash)(trial_ids.astype(np.int64))) % K  # [N]
Y = G_mat[gid_idx]    

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_tr, X_te, Y_tr, Y_te, gid_tr, gid_te = train_test_split(X, Y, gid_idx, test_size=0.2, random_state=42)

model = Ridge(alpha=10.0) 
model.fit(X_tr, Y_tr)

Y_pred = model.predict(X_te).astype(np.float32)
Y_pred /= (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)
G_norm = G_mat  # already normalized
sims = Y_pred @ G_norm.T                         # [N_test, K]
top1 = sims.argmax(axis=1)                       # predicted gallery index
top1_acc = (top1 == gid_te).mean()
mean_cos = np.mean(np.sum(Y_pred * (Y_te / (np.linalg.norm(Y_te, axis=1, keepdims=True) + 1e-8)), axis=1))

print(f"Mean cosine(pred,true): {mean_cos:.4f}")
print(f"Top-1 retrieval (synthetic): {top1_acc:.3f}")