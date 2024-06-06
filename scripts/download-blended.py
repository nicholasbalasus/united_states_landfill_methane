# The blended TROPOMI+GOSAT product is available on AWS. This script downloads
# the files for the date range of 1 January 2019 to 31 December 2022. Variables
# for the across-track and along-track dimension are also added.

import os
import re
import glob
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import multiprocessing
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
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

    # Add across- and along-track index variables. This uses a file I've
    # stored on AWS that got these variables from the operational files.
    # This only needs to be done because I didn't archive these variables
    # when making the Blended data originally.
    link = (f"https://blended-tropomi-gosat-methane.s3-us-west-2"
            f".amazonaws.com/misc/indexes.pkl")
    subprocess.run(["wget", "-q", link, "-P", "resources/"])
    idxs = pd.read_pickle("resources/indexes.pkl")
    blended_files = sorted(glob.glob(blended_dir+"/*.nc"))
    orb = [int(re.search(r'_(\d{5})_',f).groups(0)[0]) for f in blended_files]

    def add_idxs(file, idxs):
        with xr.open_dataset(file) as ds:
            for var in ["across_track_index", "along_track_index"]:
                ds[var] = (ds["pressure_interval"]*0.0 + idxs[var]).astype(int)
            for var in ["longitude", "latitude"]:
                assert np.array_equal(np.array(idxs["longitude"]),
                                      np.array(ds["longitude"].values))
            ds.to_netcdf(file)

    inputs = [(blended_files[i],
               idxs.loc[idxs["orbit"] == orb[i]].reset_index(drop=True))
              for i in range(len(orb))]
    with multiprocessing.Pool() as pool:
        pool.map(add_idxs, inputs)
        pool.close()
        pool.join()