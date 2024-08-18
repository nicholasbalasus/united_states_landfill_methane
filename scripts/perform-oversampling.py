# In this script, using inputs that describe the desired time and spatial
# bounds, we oversample the Blended TROPOMI data. The data can optionally 
# be wind-rotated around a source using HRRR winds.

import re
import sys
import json
import glob
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing
from pyproj import Geod
import geopandas as gpd
from geopy import distance
import shapely.geometry as geometry
with open("config.json", "r") as f:
    config = json.load(f)

# Input parameters to determine time and spatial range to oversample.
start_time  = pd.to_datetime(f"{sys.argv[1]} 00:00:00")
end_time    = pd.to_datetime(f"{sys.argv[2]} 00:00:00")
lon_min     = float(sys.argv[3])
lon_max     = float(sys.argv[4])
lon_res     = float(sys.argv[5])
lat_min     = float(sys.argv[6])
lat_max     = float(sys.argv[7])
lat_res     = float(sys.argv[8])
name        = sys.argv[9]
wind_rotate = (sys.argv[10] == "True")
source_lon  = float(sys.argv[11])
source_lat  = float(sys.argv[12])

if __name__ == "__main__":

    # Using multiple cores, make a grid that we can oversample to. This is 
    # defined by the latitude and longitude bounds and their resolutions.
    # The code is organized to use a core to define all of the polygons at a
    # single latitude and then merges everything together into one DataFrame.
    print(f"~~~~~~~~~~~~~~~~~ Making the oversampling grid ~~~~~~~~~~~~~~~~~")
    s = time.perf_counter()

    # Function that create polygons for our oversampling grid. It creates all
    # of the polygons at a given latitude. This allows us to use multiple cores.
    def create_polygons(lat, lat_res, lon_centers, lon_res):

        geod = Geod(ellps="WGS84")
        lats, lons, polygons, areas, heights = [],[],[],[],[]
        for lon in lon_centers:
            lats.append(lat)
            lons.append(lon)
            p_lons = [lon+lon_res/2,lon+lon_res/2,lon-lon_res/2,lon-lon_res/2]
            p_lats = [lat-lat_res/2,lat+lat_res/2,lat+lat_res/2,lat-lat_res/2]
            polygon = geometry.Polygon(zip(p_lons,p_lats))
            polygons.append(polygon)
            areas.append(geod.geometry_area_perimeter(polygon)[0])
            heights.append(distance.distance(
                (lat-lat_res/2,lon), (lat+lat_res/2,lon)).m)

        return pd.DataFrame({"lat_center": lats,
                             "lon_center": lons,
                             "polygon": polygons,
                             "area_m2": areas,
                             "height_m": heights})

    lat_centers = np.linspace(lat_min, lat_max,
                              int(np.round((lat_max-lat_min)/lat_res))+1)
    lon_centers = np.linspace(lon_min, lon_max,
                              int(np.round((lon_max-lon_min)/lon_res))+1)
    inputs = [(lat_center, lat_res, lon_centers, lon_res)
              for lat_center in lat_centers]
    with multiprocessing.Pool() as pool:
        num_processes = pool._processes
        results = pool.starmap(create_polygons, inputs)
        pool.close()
        pool.join()

    oversampling_grid = pd.concat(results, ignore_index=True)

    lat_min = oversampling_grid.lat_center.min()
    lat_max = oversampling_grid.lat_center.max()
    lon_min = oversampling_grid.lon_center.min()
    lon_max = oversampling_grid.lon_center.max()
    print(f"Latitude res/min/max  --> {lat_res}°/{lat_min}°/{lat_max}°")
    print(f"Longitude res/min/max --> {lon_res}°/{lon_min}°/{lon_max}°")
    print(f"Number of grid cells  --> {len(oversampling_grid)}")
    print(f"Time elapsed          --> {(time.perf_counter()-s)/60:.2f} min")
    print(f"Cores used            --> {num_processes} core(s)")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    # Now we will read data from all satellite files. The goal is to retrieve
    # all satellite observations that intersect our oversampling region (either
    # before or after wind-rotation) and our time domain. The filtering of 
    # observations > 5° longitude in width is to remove pixels that cross the
    # antimeridian since we can't make a valid polygon with them.
    print(f"~~~~~~~~~~~~~~~~ Collecting the satellite data ~~~~~~~~~~~~~~~~~")
    s = time.perf_counter()

    # Function that filters the blended data one netCDF file at a time.
    # Retrievals that are coastal or over water and those that are greater than
    # 5° longitude in width are filtered out.
    def filter_blended_retrievals(file, region, start_time, end_time):

        with xr.open_dataset(file) as ds:

            # Perform an initial filtering by time, antimerdian, surface.
            # Doing this first speeds up the polygon code.
            f = "%Y-%m-%dT%H:%M:%S.%fZ"
            satellite_time = pd.to_datetime(ds["time_utc"].values, format=f)
            val1 = (satellite_time >= start_time)
            val1 &= (satellite_time < end_time)
            val1 &= (ds["longitude_bounds"].values.ptp(axis=1) < 5)
            sc = ((ds["surface_classification"].values.astype("uint8") & 0x03).
                   astype(int))
            val1 &= ~(sc == 1) # Over water
            val1 &= ~((sc == 3) | ((sc == 2) & (ds["chi_square_SWIR"].values
                                                > 20000))) # Coastal

            # Make a GeoSeries of the satellite polygons.
            lat_corners = ds["latitude_bounds"].values[val1]
            lon_corners = ds["longitude_bounds"].values[val1]
            sat_polygons = gpd.GeoSeries(
                        [geometry.Polygon(zip(lon_corners[i],lat_corners[i]))
                         for i in range(len(lat_corners))])

            # Filter out retrievals that don't intersect our region.
            val2 = sat_polygons.intersects(region)

            # Calculate the area in square meters
            # Also get the orbit number for each observation
            polygons = list(sat_polygons[val2])
            geod = Geod(ellps="WGS84")
            areas, orbits = [],[]
            for i in range(len(polygons)):
                areas.append(geod.geometry_area_perimeter(polygons[i])[0])
                orbits.append(re.search(r'_(\d{5})_', file).groups(0)[0])

            # Make a DataFrame that only includes our valid values
            satellite_data = pd.DataFrame(
                {"lat_center": ds["latitude"].values[val1][val2],
                 "lon_center": ds["longitude"].values[val1][val2],
                 "orbit_number": orbits,
                 "time_utc": satellite_time[val1][val2],
                 "polygon": polygons,
                 "polygon_area_m2": areas,
                 "xch4_ppb": (ds["methane_mixing_ratio_blended_destriped"]
                              .values[val1][val2]),
                 "dry_air_mol_m2": np.sum(ds["dry_air_subcolumns"]
                                          .values[val1][val2],axis=1),
                 "albedo": ds["surface_albedo_SWIR"].values[val1][val2],
                 "aerosol_size": ds["aerosol_size"].values[val1][val2],
                })

        return satellite_data

    # List of blended files
    files = sorted(glob.glob(config['blended_dir'] + "/*.nc"))

    # Define our oversampling region.
    oversampling_region = geometry.Polygon(zip(
                            [lon_min-lon_res/2,lon_min-lon_res/2,
                             lon_max+lon_res/2,lon_max+lon_res/2],
                            [lat_min-lat_res/2,lat_max+lat_res/2,
                             lat_max+lat_res/2,lat_min-lat_res/2]))

    # Optionally, wind rotate the pixels around a source location.
    if not wind_rotate:

        # Using multiple cores, collect all of the satellite observations
        # that intersect our oversampling region into a single DataFrame.
        with multiprocessing.Pool() as pool:
            num_processes = pool._processes
            inputs = [(file, oversampling_region, start_time, end_time)
                      for file in files]
            results = pool.starmap(filter_blended_retrievals, inputs)
            pool.close()
            pool.join()
        satellite_df = pd.concat(results, ignore_index=True)

    else:

        # If we are wind-rotating, collect all observations that intersect
        # our oversampling region with a buffer equal to the max region width.
        # This is so that we get all observations that could intersect our
        # region following the wind rotation.
        region = oversampling_region.buffer(
            np.max([(lat_max - lat_min), (lon_max - lon_min)]))
        with multiprocessing.Pool() as pool:
            inputs = [(file, region, start_time, end_time)
                      for file in files]
            results = pool.starmap(filter_blended_retrievals, inputs)
            pool.close()
            pool.join()
        satellite_df = pd.concat(results, ignore_index=True)

        # Now, perform the wind rotation on the remaining polygons.            
        # Use a sample HRRR file to get the x,y indexes of the grid cells
        # within 5 km of the source coordinates.
        file = f"{config['hrrr_dir']}/hrrr_2019-01-01.nc"
        with xr.open_dataset(file) as ds:
            hrrr_lons = ds["lon"].values - 360
            hrrr_lats = ds["lat"].values

        mask = (np.abs(hrrr_lats - source_lat) < 0.15) &\
               (np.abs(hrrr_lons - source_lon) < 0.15)
        nearby_hrrr_lons = hrrr_lons[mask]
        nearby_hrrr_lats = hrrr_lats[mask]
        distances = np.zeros((len(nearby_hrrr_lats)))
        for idx in range(len(distances)):
            hrrr_grid_cell = (nearby_hrrr_lats[idx], nearby_hrrr_lons[idx])
            source = (source_lat, source_lon)
            distances[idx] = distance.distance(hrrr_grid_cell, source).km
        ys,xs = [],[]
        for idx in np.where(distances < 5)[0]:
            y,x = np.where(hrrr_lats == nearby_hrrr_lats[idx])
            ys.append(y[0])
            xs.append(x[0])

        # Extract the U and V winds at every hour.
        # Average over all grid cells that were within 5 km of the source.
        t = np.array((), dtype="datetime64[ns]")
        u = np.array((), dtype="float32")
        v = np.array((), dtype="float32")
        files = sorted(glob.glob(f"{config['hrrr_dir']}/*.nc"))
        for file in files:
            with xr.open_dataset(file) as ds:
                ds_mean = ds.sel(y=ys, x=xs).mean(dim=["y","x"])
                t = np.append(t, ds_mean["time"].values)
                u = np.append(u, ds_mean["u"].values)
                v = np.append(v, ds_mean["v"].values)

        # For each satellite observation, average over the previous 3 hours.
        for idx in satellite_df.index:
            base = satellite_df.loc[idx,"time_utc"].floor(freq="h")
            three_hours = [np.datetime64(base-pd.Timedelta(i,"h"))
                           for i in range(0,3)]
            mask = np.isin(t,three_hours)
            assert np.sum(mask) == 3
            satellite_df.loc[idx,"u"] = u[mask].mean()
            satellite_df.loc[idx,"v"] = v[mask].mean()

        # Calculate the degrees to rotate each satellite observation to align
        # the winds at that time with the (1,0) vector (westerly winds).
        def angle_to_rotate(wind_north, wind_east):
            angle_radians = np.arctan2(0.0, 1.0) - \
                            np.arctan2(wind_north, wind_east)
            angle_degrees = np.degrees(angle_radians)
            angle_degrees %= 360
            return angle_degrees
        thetas = angle_to_rotate(satellite_df["v"], satellite_df["u"])

        # Rotate the polygons around the source using multiple cores.
        def rotate_polygons(polygon_chunk, theta_chunk, source_lon, source_lat):
            rotated_polygons = []
            for idx,polygon in enumerate(polygon_chunk):
                rotated_polygons.append(
                    gpd.GeoSeries(polygon)
                    .rotate(theta_chunk[idx], origin=(source_lon,source_lat))
                    .iloc[0])
            return rotated_polygons

        with multiprocessing.Pool() as pool:
            num_processes = pool._processes
            polygon_chunks = np.array_split(np.array(satellite_df["polygon"]),
                                            num_processes)
            theta_chunks = np.array_split(np.array(thetas), num_processes)
            inputs = [(polygon_chunks[idx], theta_chunks[idx], 
                       source_lon, source_lat)
                      for idx in range(len(polygon_chunks))]
            results = pool.starmap(rotate_polygons, inputs)
            pool.close()
            pool.join()

        # Replace the polygons in satellite_df with the rotated polygons.
        satellite_df["polygon"] = [item for sublist in results
                                   for item in sublist]
        
        # Drop all rotated polygons that do not intersect oversampling region.
        sat_polygons = gpd.GeoSeries(satellite_df["polygon"])
        valid_idx = sat_polygons.intersects(oversampling_region)
        satellite_df = satellite_df.loc[valid_idx].reset_index(drop=True)

    # Drop anomalous observations
    max = satellite_df["xch4_ppb"].mean() + 3*satellite_df["xch4_ppb"].std()
    min = satellite_df["xch4_ppb"].mean() - 3*satellite_df["xch4_ppb"].std()
    valid_idx = ((satellite_df["xch4_ppb"] > min) &
                 (satellite_df["xch4_ppb"] < max))
    satellite_df = satellite_df.loc[valid_idx].reset_index(drop=True)

    # Drop low wind speed observations
    if wind_rotate:
        wind_speed = np.sqrt(satellite_df["u"]**2 + satellite_df["v"]**2)
        valid_idx = wind_speed > 1 # [m/s]
        satellite_df = satellite_df.loc[valid_idx].reset_index(drop=True)

    print(f"Number of observations   --> {len(satellite_df)}")
    unique_days = len(satellite_df["time_utc"].dt.date.unique())
    print(f"Unique number of days    --> {unique_days}")
    print(f"Minimum observation time --> {satellite_df.time_utc.min()}")
    print(f"Maximum observation time --> {satellite_df.time_utc.max()}")
    print(f"Wind rotation            --> {wind_rotate}")
    if wind_rotate is True:
        print(f"Rotating around lat/lon  --> {source_lat}°/{source_lon}°")
    print(f"Time elapsed             --> {(time.perf_counter()-s)/60:.2f} min")
    print(f"Cores used               --> {num_processes} core(s)")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    # With the data now filtered, we can perform the oversampling.
    # We perform a tessellation-style regridding following Zhu et al. (2017).
    print(f"~~~~~~~~~~~~~~~~~ Performing the oversampling ~~~~~~~~~~~~~~~~~~")
    s = time.perf_counter()

    # Performs oversampling for one chunk of grid cells on oversampling grid.
    def oversample(chunk, satellite_df):

        # Copy the chunk of the oversampling grid.
        oversampled_data_chunk = chunk.copy(deep=True)

        # Convert polygons to GeoSeries
        satellite_polygons = gpd.GeoSeries(satellite_df["polygon"])

        # Determine which variables to oversample
        do_not_oversample = ["lat_center","lon_center","orbit_number",
                             "time_utc","polygon","polygon_area_m2","u","v"]
        vars = [v for v in satellite_df.keys() if v not in do_not_oversample]

        # Loop through each grid cell on the oversampling grid.
        for j in oversampled_data_chunk.index:

            # Only use the satellite observations that intersect this grid cell.
            valid_idx = satellite_polygons.intersects(
                oversampled_data_chunk.loc[j,"polygon"])
            
            # If we don't have any intersecting pixels, leave as NaN
            if np.sum(valid_idx) == 0:
                continue

            # Calculate a weighted average. The weights for each satellite
            # observations (w_i) are proportional to the ratio of the area
            # of satellite observation i that intersects with polygon j (S_ij)
            # to the total area of satellite observation i (A_i).
            geod = Geod(ellps="WGS84")
            intersections = (satellite_polygons.loc[valid_idx]
                             .intersection(oversampled_data_chunk.loc
                                           [j,"polygon"]))
            S_ij = np.array([abs(geod.geometry_area_perimeter(polygon)[0])
                             for polygon in intersections]) # [m2]
            A_i = satellite_df["polygon_area_m2"].loc[valid_idx] # [m2]
            w_i = S_ij/A_i

            # Calculate the weighted average and observational density.
            if np.sum(w_i) != 0.0:

                for var in vars:
                    omega_i = satellite_df.loc[valid_idx,var]
                    oversampled_data_chunk.loc[j,var] = (
                        np.sum(w_i*omega_i)/np.sum(w_i))
                    
                oversampled_data_chunk.loc[j,"layers_satellite_pixels"] = (
                    np.sum(S_ij)/oversampled_data_chunk.loc[j,"area_m2"])

        return oversampled_data_chunk

    with multiprocessing.Pool() as pool:
        number_of_chunks = pool._processes
        chunk_idxs = np.array_split(oversampling_grid.index, number_of_chunks)
        inputs = [(oversampling_grid.loc[chunk_idx], satellite_df)
                  for chunk_idx in chunk_idxs]
        results = pool.starmap(oversample, inputs)
        pool.close()
        pool.join()
    oversampled_data = pd.concat(results, ignore_index=True)

    # Put average winds at the source as an attribute in the DataFrame.
    if wind_rotate is True:
        wind_speed = np.sqrt(satellite_df["u"]**2 + satellite_df["v"]**2)
        harmonic_mean_wind_speed = len(wind_speed)/np.sum(1/wind_speed)
        oversampled_data.attrs["wind_speed_m_s"] = harmonic_mean_wind_speed
        oversampled_data.attrs["U_m_s"] = satellite_df["u"]
        oversampled_data.attrs["V_m_s"] = satellite_df["v"]
        oversampled_data.attrs["num_days"] = unique_days

    # Put times and orbit numbers as an attribute in the DataFrame.
    oversampled_data.attrs["observation_times"] = satellite_df["time_utc"]
    oversampled_data.attrs["orbit_numbers"] = satellite_df["orbit_number"]

    oversampled_data.to_pickle(f"{name}.pkl")
    print(f"Time elapsed --> {(time.perf_counter()-s)/60:.2f} min")
    print(f"Cores used   --> {number_of_chunks} core(s)")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")