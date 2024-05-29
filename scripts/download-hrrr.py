# On AWS, there is an archive of HRRR forecasts. To create a data record,
# the 0th hour forecast from each hour is used. The u and v winds averaged
# across the lowest 100 hPa are assembled into an xarray dataset with
# dimensions of time, y, and x and data variables of u, v, latitude,
# and longitude. This scipt can get the data for an arbitrary range of 
# dates that are fed as inputs.

import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing
import subprocess
import os
import sys

# The config file specifies the directory to store the files in.
# Specify "dont_write_bytecode" to avoid __pycache__ creation.
sys.dont_write_bytecode = True
from config import hrrr_dir

# The two inputs to this script should be the start date ("2019-01-01")
# and the (exclusive) end date ("2019-02-01"). The examples in parentheses
# would get wind for all of January 2019 (a resulting file of ~11 GB).
start_date = sys.argv[1]
end_date = sys.argv[2]
print(f"Getting HRRR winds for [{start_date},{end_date})\n")

# Function that gets winds from the grib file then deletes it
def get_winds(hour,day,month,year,lon,lat):

    grib_file = f"{tmp_dir}/hrrr.t{str(hour).zfill(2)}z.wrfnatf00.grib2"

    if os.path.exists(grib_file):

        # Retrieve the surface pressure
        with xr.open_dataset(grib_file, engine="cfgrib",\
        backend_kwargs={"filter_by_keys": {"typeOfLevel": \
        "surface", "stepType": "instant"}}) as ds:
            surface_pressure = ds["sp"].values
        os.remove(grib_file+".923a8.idx")

        # Average the winds over the lowest 100 hPa of the atmosphere
        # Pressure-weight this average
        with xr.open_dataset(grib_file, engine="cfgrib",\
        backend_kwargs={"filter_by_keys": {"typeOfLevel": \
        "hybrid"}}) as ds:
            mask = (ds["pres"].values > (surface_pressure - 10000))
            mask_u = np.ma.MaskedArray(ds["u"].values, mask=~mask)
            u = np.ma.average(mask_u, weights=ds["pres"].values, axis=0).filled()
            mask_v = np.ma.MaskedArray(ds["v"].values, mask=~mask)
            v = np.ma.average(mask_v, weights=ds["pres"].values, axis=0).filled()
            assert np.array_equal(lon, ds["longitude"].values)
            assert np.array_equal(lat, ds["latitude"].values)
        os.remove(grib_file+".923a8.idx")

        # Remove the file
        os.remove(grib_file)
            
    else:
        u = np.zeros((1059,1799))*np.nan
        v = np.zeros((1059,1799))*np.nan
        print(f"File is missing for {year}-{month}-{day} {hour}:00:00",
              flush=True)

    t = pd.to_datetime(f"{year}-{month}-{day} {hour}:00:00")

    return u, v, t

if __name__ == "__main__":

    # Using pandas, we will make lists of every day that we need to download.
    years, months, days = [], [], []
    current_date = pd.to_datetime(start_date)
    while current_date != pd.to_datetime(end_date):
        years.append(current_date.strftime("%Y"))
        months.append(current_date.strftime("%m"))
        days.append(current_date.strftime("%d"))
        current_date += pd.Timedelta(days=1)

    # Each HRRR forecast will have the same y (1059) and x (1059) size.
    # We'll collect arrays of the averaged u and v winds into these arrays.
    # The third dimension is time which is one hour per day.
    u = np.zeros((1059,1799,len(days)*24),dtype="float32")*np.nan
    v = np.zeros((1059,1799,len(days)*24),dtype="float32")*np.nan
    t = np.zeros((len(days)*24),dtype="object")*np.nan

    # From the Jan 1, 2019 00:00:00 file, we will get the values for
    # x, y, lon, and lon. We check to make sure these longitudes and
    # latitudes match all of the files that we open.
    tmp_dir = f"{hrrr_dir}/hrrr_tmp_{start_date}"
    os.makedirs(tmp_dir, exist_ok=True)
    link = (f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr."
            f"20190101/conus/hrrr.t00z.wrfnatf00.grib2")
    subprocess.run(["wget", "-q", "-P", tmp_dir, link])
    with xr.open_dataset(f"{tmp_dir}/hrrr.t00z.wrfnatf00.grib2", 
                         engine="cfgrib",backend_kwargs={"filter_by_keys":\
                         {"typeOfLevel": "hybrid"}}) as ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        x   = ds["x"].values
        y   = ds["y"].values
    os.remove(f"{tmp_dir}/hrrr.t00z.wrfnatf00.grib2")
    os.remove(f"{tmp_dir}/hrrr.t00z.wrfnatf00.grib2.923a8.idx")

    # The files are downloaded and processed one day at a time. The AWS
    # links are predictable, so the 24 links for each day are assembled
    # into a txt file and then downloaded with multiple cores with wget.
    idx = 0
    for day,month,year in zip(days,months,years):

        assert len(os.listdir(tmp_dir)) == 0
        print(f"Downloading HRRR 0th hour forecasts for {year}-{month}-{day}",
              flush=True)
        with open(f"{tmp_dir}/links.txt", "w") as f:
            for hour in range(24):
                f.write(f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr."
                        f"{year}{month}{day}/conus/"
                        f"hrrr.t{str(hour).zfill(2)}z.wrfnatf00.grib2\n")

        with open(f"{tmp_dir}/links.txt", "r") as file:
            subprocess.run(["xargs", "-n", "1", "-P", "24", "wget", "-q",
                            "-P", tmp_dir], stdin=file)
        os.remove(f"{tmp_dir}/links.txt")

        # We now have all of the grib2 files for the day. It is usually 24,
        # but if a file is missing, the u and v values will be left as nan.
        # If the file exists, we will average the winds across the lowest
        # 100 hPa and fill in our u and v numpy arrays.
        inputs = [(hour, day, month, year, lon, lat) for hour in range(24)]
        with multiprocessing.Pool(24) as pool:
            results = pool.starmap(get_winds,inputs)
            pool.close()
            pool.join()

        for u_, v_, t_ in results:
            u[:,:,idx] = u_
            v[:,:,idx] = v_
            t[idx] = t_
            idx += 1

    # Drop the NaN values. This occurs when a HRRR forecast is missing.
    mask = ~np.isnan(u[0,0,:])
    u = u[:,:,mask]
    v = v[:,:,mask]
    t = t[mask]

    # Now we have numpy arrays that can be assembled into a dataset.
    os.rmdir(tmp_dir)
    hrrr_winds = xr.Dataset(data_vars=dict(
                            u=(["y","x","t"], u), v=(["y","x","t"], v),
                            lat=(["y","x"], lat), lon=(["y","x"], lon)),
                            coords=dict(
                            y=(["y"], y), x=(["x"], x),t=(["t"], t))
                            )
    hrrr_winds.to_netcdf(f"{hrrr_dir}/hrrr_winds_{start_date}.nc")