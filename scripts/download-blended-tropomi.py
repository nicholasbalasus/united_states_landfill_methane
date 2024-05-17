# The blended TROPOMI+GOSAT product is available on AWS. This script downloads
# the files for the date range of 1 January 2019 to 31 December 2022.

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import multiprocessing
import sys

# The config file specifies the directory to store the files in.
# Specify "dont_write_bytecode" to avoid __pycache__ creation.
sys.dont_write_bytecode = True
sys.path.append('..')
from config import blended_dir

if __name__ == "__main__":
    
    # Equivalent of --no-sign-request for boto3
    s3 = None
    def initialize():
        global s3
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    initialize()

    # Collect names of paths
    months = ([f"{y}-{str(m).zfill(2)}" for m in range(1,13)
               for y in range(2019,2023)])
    bucket_name = "blended-tropomi-gosat-methane"
    s3_paths = []
    for month in months:
        Prefix=(f"data/{month}/")
        for k in s3.list_objects(Bucket=bucket_name, Prefix=Prefix)["Contents"]:
            s3_paths.append(k["Key"])

    # Download the files using all 112 cores
    os.makedirs(blended_dir, exist_ok=True)

    def download_from_s3(s3_path):
        file = os.path.basename(s3_path)
        local_file_path = os.path.join(blended_dir,file)
        s3.download_file(bucket_name, s3_path, local_file_path)

    with multiprocessing.Pool(112, initialize) as pool:
        pool.map(download_from_s3, s3_paths)
        pool.close()
        pool.join()