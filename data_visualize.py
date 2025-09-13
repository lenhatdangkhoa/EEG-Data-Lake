import boto3
import os
import mne 
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# AWS Configuration
AWS_REGION = "us-east-1"
BUCKET="eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze/"
SILVER_PREFIX = "silver/"

s3 = boto3.client('s3', region_name=AWS_REGION)

eeg_example_file = "s3://eeg-data-lake-khoa/bronze/MindBigData_Imagenet_Insight_n00007846_112420_1_1479.csv"

df = pd.read_csv(eeg_example_file, header=None)

print(df.head())
print(df.shape)
print(len(df.columns))
print(df.columns)

df_silver = df.set_index(0).T   # first col = channel names, then transpose
df_silver.index.name = "timepoint"
df_silver = df_silver.dropna()

def bandpass_filter(data, low=1, high=40, fs=128, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data)

df_silver = df_silver.apply(lambda x: bandpass_filter(x.values), axis=0)

scaler = StandardScaler()
df_silver[df_silver.columns] = scaler.fit_transform(df_silver)
print(df_silver.head())
plt.figure(figsize=(12,6))

for col in df_silver.columns:
    plt.plot(df_silver.index, df_silver[col], label=col)

plt.xlabel("Time (samples)")
plt.ylabel("EEG amplitude (standardized)")
plt.title("EEG signals over time")
plt.legend()
plt.savefig("eeg_signals_random_run_silver.png")
