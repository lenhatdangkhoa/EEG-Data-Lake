import boto3
import os
import mne 
import numpy as np
import scipy

# AWS Configuration
AWS_REGION = "us-east-1"
BUCKET="eeg-data-lake-khoa"
BRONZE_PREFIX = "bronze/"
SILVER_PREFIX = "silver/"

s3 = boto3.client('s3', region_name=AWS_REGION)


