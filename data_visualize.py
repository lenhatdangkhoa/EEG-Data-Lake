import boto3
import os
import mne 
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# AWS Configuration
AWS_REGION = "us-east-1"
BUCKET="eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze/"
SILVER_PREFIX = "silver/"

s3 = boto3.client('s3', region_name=AWS_REGION)

eeg_example_file = "s3://eeg-data-lake-khoa/bronze/MindBigData_Imagenet_Insight_n00007846_112420_1_1479.csv"

df = pd.read_csv(eeg_example_file)

print(df.head())
print(df.shape)
print(len(df.columns))

n_channels, n_times = df.shape
freq = 128 # 128Hz
t = np.arange(n_times) / freq
ch_names = ["AF4", "T7", "T8", "Pz"]

plt.figure(figsize=(15, 7))
for i, ch in enumerate(ch_names):
    plt.plot(t, df[ch], label=ch)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("EEG Signals from 4 Channels")
plt.legend()
plt.show()