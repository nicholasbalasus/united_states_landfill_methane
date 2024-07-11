import os
import sys
import glob
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
sys.dont_write_bytecode = True
sys.path.append("scripts")
import scraper
with open("config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":

    # (1) Oversample around each landfill for each year
    count = 1
    for landfill in config["landfills"].keys():

        source_lon = config["landfills"][landfill]["lon"]
        source_lat = config["landfills"][landfill]["lat"]

        # Oversample for a 0.6° x 0.6° domain centered on the source
        # Resolution is 0.01° x 0.01° (~1 km)
        lon_min = float(f"{round(source_lon, 2) - 0.3:.2f}")
        lon_max = float(f"{round(source_lon, 2) + 0.3:.2f}")
        lat_min = float(f"{round(source_lat, 2) - 0.3:.2f}")
        lat_max = float(f"{round(source_lat, 2) + 0.3:.2f}")
        lat_res, lon_res = 0.01, 0.01

        # Oversample for 2019, 2020, 2021, 2022, 2023
        for year in range(2019,2024):
            name = f"notebooks/output/{landfill}-{year}"
            start_time = f"{year}-01-01"
            end_time = f"{year+1}-01-01"
            wind_rotate = True
            
            # Submit the job using SLURM, waiting if this is the last job
            command = [
                "sbatch", "-p", "sapphire,huce_ice", "-t", "30",
                "--mem", "128000", "-c", "64", "--wrap", (f"source ~/.bashrc; "
                f"micromamba activate ldf_env; "
                f"python scripts/perform-oversampling.py "
                f"{start_time} {end_time} {lon_min} {lon_max} {lon_res} "
                f"{lat_min} {lat_max} {lat_res} {name} {wind_rotate} "
                f"{source_lon} {source_lat}")
            ]
            if count == 20:
                command.insert(1, "--wait")
            subprocess.run(command)
            count += 1

    # (2) Scrape data from FLIGHT
    assembled = {}
    for landfill in config["landfills"].keys():
        assembled[landfill] = {}
        # This function returns all years but we only want 2019 and onward
        id = config["landfills"][landfill]["id"]
        assembled[landfill]["data"] = (scraper.scrape_flight_ghgrp(id).iloc[9:]
                                       .reset_index(drop=True))

    # (3) Perform emissions estimates
    for landfill in config["landfills"].keys():
        
        # DataFrame to gather information in
        source_lon = config["landfills"][landfill]["lon"]
        source_lat = config["landfills"][landfill]["lat"]
    
        for idx,year in enumerate(range(2019,2024)):

            # Read oversampled data
            df = pd.read_pickle(f"notebooks/output/{landfill}-{year}.pkl")
            gdf = gpd.GeoDataFrame(df)
            gdf = gdf.set_geometry("polygon")
            gdf["polygon"] = gdf.buffer(0.001)
            density_mask = gdf["layers_satellite_pixels"] >= 40
            gdf = gdf[density_mask].reset_index(drop=True)

            # Area that CSF is performed for
            lon_diff = gdf["lon_center"]+0.005-source_lon
            lat_diff = np.abs(gdf["lat_center"]-source_lat)
            gdf["csf"] = ((lon_diff > 0) & (lon_diff < 0.1) & (lat_diff < 0.1))
            bkg_xch4 = np.mean(gdf.loc[~gdf["csf"], "xch4_ppb"])
            gdf["delta_xch4_ppb"] = gdf["xch4_ppb"] - bkg_xch4
            assembled[landfill][f"oversampled-{year}"] = gdf

            # Calculate emissions best estimate
            Q_best = []
            for lon in gdf["lon_center"].unique():
                subset = gdf.loc[(gdf["lon_center"] == lon) &
                                (gdf["csf"])].reset_index(drop=True)
                if len(subset) > 0:
                    dry_air = np.array(subset["dry_air_mol_m2"])
                    height = np.array(subset["height_m"])
                    delta_xch4 = subset["delta_xch4_ppb"]
                    delta_omega = delta_xch4*1e-9*dry_air*16.04*1e-3 # [kg/m2]
                    delta_omega_times_height = delta_omega*height # [kg/m]
                    transect = delta_omega_times_height.sum() # [kg/m]
                    U = df.attrs["wind_speed_m_s"]*60*60 # [m/h]
                    Q_best.append(transect*U*24*365*1e-6) # [Gg/yr]
            Q_best = np.mean(Q_best)
            assembled[landfill]["data"].loc[idx,"Emis_Satellite"] = Q_best

            # Calculate 95% CI on emissions estimate
            Q_uncert = np.zeros((10000))*np.nan
            for idy in range(len(Q_uncert)):
                emis = np.array(())
                for lon in gdf["lon_center"].unique():
                    subset = gdf.loc[(gdf["lon_center"] == lon) &
                                    (gdf["csf"])].reset_index(drop=True)
                    if len(subset) > 0:
                        dry_air = np.array(subset["dry_air_mol_m2"])
                        height = np.array(subset["height_m"])
                        bkg_xch4 = np.array(gdf.loc[~gdf["csf"], "xch4_ppb"])
                        bkg_xch4 = np.mean(np.random.choice(bkg_xch4,
                                                            (len(bkg_xch4),)))
                        delta_xch4 = subset["xch4_ppb"] - bkg_xch4
                        delta_omega = (delta_xch4*1e-9*dry_air*
                                       16.04*1e-3) # [kg/m2]
                        delta_omega_times_height = delta_omega*height # [kg/m]
                        transect = delta_omega_times_height.sum() # [kg/m]
                        U = np.array((df.attrs["U_m_s"]**2 + 
                                      df.attrs["V_m_s"]**2)**0.5) # [m/s]
                        U = np.mean(np.random.choice(U, (len(U),)))*3600 # [m/h]
                        e = transect*U*24*365*1e-6 # [Gg/yr]
                        emis = np.append(emis, e)
                Q_uncert[idy] = np.mean(np.random.choice(emis, (len(emis),)))
            Q_u = np.percentile(Q_uncert, 97.5)
            Q_l = np.percentile(Q_uncert, 2.5)
            assembled[landfill]["data"].loc[idx,"Emis_Satellite_Upper"] = Q_u
            assembled[landfill]["data"].loc[idx,"Emis_Satellite_Lower"] = Q_l

    # (4) Save assembled data and cleanup
    files = glob.glob("notebooks/output/*.pkl")
    for file in files:
        os.remove(file)
    with open("notebooks/output/assembled.pkl", "wb") as handle:
        pickle.dump(assembled, handle, protocol=pickle.HIGHEST_PROTOCOL)