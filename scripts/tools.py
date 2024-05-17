import os
import glob
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cmcrameri import cm

def oversampling(source_lon, source_lat):
    """ Oversamples with wind rotation around a given source (2019-2022)
    Parameters
    ----------
    source_lon : float
        Longitude of the source to be rotated around
    source_lat : float
        Latitude of the source to be rotated around
    
    Returns
    -------
    none : none
        Saves i.pkl (for i = 1-4) files of the oversampled data
    """

    # Oversample for a 0.6° x 0.6° domain centered on the source
    # Resolution is 0.01° x 0.01° (~1 km)
    lon_min = float(f"{round(source_lon, 2) - 0.3:.2f}")
    lon_max = float(f"{round(source_lon, 2) + 0.3:.2f}")
    lat_min = float(f"{round(source_lat, 2) - 0.3:.2f}")
    lat_max = float(f"{round(source_lat, 2) + 0.3:.2f}")
    lat_res, lon_res = 0.01, 0.01

    # Oversample for 2019, 2020, 2021, 2022
    for idx,year in enumerate(range(2019,2023)):
        name = f"{idx+1}"
        start_time = f"{year}-01-01"
        end_time = f"{year+1}-01-01"
        wind_rotate = True

        # Submit the job using SLURM
        subprocess.run([
            "sbatch", "-p", "sapphire,serial_requeue", "-t", "180",
            "--mem", "128000", "-c", "48", "--wrap", (f"source ~/.bashrc; "
            f"micromamba activate ldf_env; "
            f"python ../scripts/perform-oversampling.py "
            f"{start_time} {end_time} {lon_min} {lon_max} {lon_res} {lat_min} "
            f"{lat_max} {lat_res} {name} {wind_rotate} "
            f"{source_lon} {source_lat}")
        ])
        
def masks(name, source_lon, source_lat):
    """ Determines plume and background masks for oversampled data
    Parameters
    ----------
    name : str
        Prefix of the .pkl file with the oversampled data
    source_lon : float
        Longitude of the source to be rotated around
    source_lat : float
        Latitude of the source to be rotated around
    
    Returns
    -------
    oversampled_data : pandas.core.frame.DataFrame
        DataFrame of the oversampled data with boolean masks added
    """
    
    # Make 2D arrays of lat, lon, xch4
    oversampled_data = pd.read_pickle(f"{name}.pkl")
    lon_2d, lat_2d = np.meshgrid(
                    oversampled_data["lon_center"].unique(),
                    oversampled_data["lat_center"].unique())
    xch4_2d = np.zeros_like(lon_2d)*np.nan
    layers_2d = np.zeros_like(lon_2d)*np.nan
    for iy,ix in np.ndindex(lon_2d.shape):
        subset = oversampled_data[
            (oversampled_data.lat_center == lat_2d[iy,ix]) &\
            (oversampled_data.lon_center == lon_2d[iy,ix])]
        assert len(subset) == 1
        xch4_2d[iy,ix] = subset.iloc[0]["xch4_ppb"]
        layers_2d[iy,ix] = subset.iloc[0]["layers_satellite_pixels"]
    xch4_2d[layers_2d < 40] = np.nan

    # First, take 98th percentile
    plume_mask = xch4_2d > np.nanpercentile(xch4_2d, 98)

    # Run the plume mask through a 3 x 3 median filter
    plume_mask = ndimage.median_filter(plume_mask, 3)

    # Finally, only keep pixels in continuous chunk closest to the source
    labeled_plume, num_features = ndimage.label(plume_mask)
    lat_dist = [np.min(np.abs(lon_2d[labeled_plume == feature] - source_lon))
                for feature in range(1,num_features+1)]
    plume_mask = np.where((labeled_plume == np.argmin(lat_dist) + 1),
                           plume_mask, False)

    # Bkg mask is 0.2° x 0.19° region that is 0.1° upwind of plume mask/source
    i = np.argmin(lon_2d[plume_mask])
    ref_lon = min(lon_2d[plume_mask][i],
                  lon_2d.flat[np.abs(lon_2d - source_lon).argmin()])
    ref_lat = lat_2d.flat[np.abs(lat_2d - source_lat).argmin()]
    bkg_mask = np.zeros_like(plume_mask, dtype=bool)
    start_iy,start_ix = np.where((lon_2d == ref_lon) &\
                                 (lat_2d == ref_lat))

    for iy,ix in np.ndindex(plume_mask.shape):
        if (abs(ix-start_ix) <= 24
            and abs(ix-start_ix) >= 5
            and lon_2d[iy,ix] < ref_lon
            and abs(iy-start_iy) <= 9):
            bkg_mask[iy,ix] = True

    # Reshape the plume mask to 1D for the DataFrame
    plume_mask_1d = np.zeros_like(oversampled_data.index, dtype=bool)
    bkg_mask_1d = np.zeros_like(oversampled_data.index, dtype=bool)
    for iy,ix in np.ndindex(plume_mask.shape):
        subset = oversampled_data[
            (oversampled_data.lat_center == lat_2d[iy,ix]) & 
            (oversampled_data.lon_center == lon_2d[iy,ix])].index
        assert len(subset) == 1
        if plume_mask[iy,ix]:
            plume_mask_1d[subset] = True
        if bkg_mask[iy,ix]:
            bkg_mask_1d[subset] = True

    oversampled_data["plume"] = plume_mask_1d
    oversampled_data["bkg"] = bkg_mask_1d

    return oversampled_data

def emissions(source_lon, source_lat):
    """ Plots oversampled data and calculates emissions for four years
    Parameters
    ----------
    source_lon : float
        Longitude of the source to be rotated around
    source_lat : float
        Latitude of the source to be rotated around
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with four subplots of oversampled data
    E_mean : numpy.ndarray
        Best estimates of the emissions each of the four years
    E_min : numpy.ndarray
        Lower bound of the 95th CI for emissions
    E_max : numpy.ndarray
        Upper bound of the 95th CI for emissions 
    """

    # Make sure all oversampling jobs are finished
    files_exists = [os.path.exists(f"{idx}.pkl") for idx in range(0,4)]
    if not all(file_exists):
        print("Error - not all oversampling jobs have finished.")
        return

    # Initialize four returns of this function
    fig, axs = plt.subplots(1, 4, figsize=(12, 5),
                            subplot_kw={"projection": ccrs.PlateCarree()})
    E_mean = np.zeros((4))
    E_min  = np.zeros((4))
    E_max  = np.zeros((4))

    # Consistent formatting for each of the four subplots
    extent = [float(f"{round(source_lon, 2) - 0.3:.2f}")-0.005,
              float(f"{round(source_lon, 2) + 0.3:.2f}")+0.005,
              float(f"{round(source_lat, 2) - 0.3:.2f}")-0.005,
              float(f"{round(source_lat, 2) + 0.3:.2f}")+0.005,]

    # Loop through each of the four years
    # Plot each year and calculate the emissions
    for idx,year in enumerate(range(2019,2023)):

        # Read in the oversampled data and add plume/background masks
        name = idx + 1
        df = masks(name, source_lon, source_lat)

        # Convert to a GeoDataFrame for easy plotting
        # Plot ∆XCH4 [ppb] which is above background
        # Also plot plume and background masks
        gdf = gpd.GeoDataFrame(df)
        gdf = gdf.set_geometry("polygon")
        gdf["polygon"] = gdf.buffer(0.001)
        density_mask = gdf["layers_satellite_pixels"] >= 40
        bkg_xch4 = (gdf[gdf["bkg"]]["xch4_ppb"].mean()) # [ppb]
        gdf["delta_xch4"] = gdf["xch4_ppb"] - bkg_xch4
        gdf[density_mask].plot(ax=axs[idx], column="delta_xch4",
            cmap=cm.navia, legend=True, vmin=-6, vmax=6,
            legend_kwds={"label":"$\Delta$XCH$_4$ [ppb]",
                         "pad": 0.04, "extend":"both"})
        axs[idx].scatter(source_lon, source_lat, marker="o", 
                         edgecolor="k", facecolor="None")
        gdf[gdf["plume"]].dissolve("plume").plot(ax=axs[idx],color="None",
                                                 linewidth=1.5,alpha=0.5)
        gdf[gdf["bkg"]].dissolve("bkg").plot(ax=axs[idx],color="None",alpha=0.5,
                                             linewidth=1.5,linestyle="dashdot")

        # Calculate emissions using means
        gdf["delta_omega"] = ((gdf["xch4_ppb"]-bkg_xch4)*1e-9*
                              gdf["dry_air_mol_m2"]*16.04*1e-3) # [kg/m2]
        gdf["delta_omega_times_height"] = (gdf["delta_omega"]*
                                           gdf["height_m"]) # [kg/m]
        U = df.attrs["wind_speed_m_s"]*60*60 # [m/h]
        sums = (gdf[gdf["plume"]].groupby("lon_center")
                [["delta_omega_times_height"]].sum()) # [kg/m]
        E_means[idx] = U*sums.values.flatten()*24*365*1e-6 # [kt/yr]

        # Calculate emission uncertainty using bootstrap
        E_i = np.zeros((10000)) * np.nan
        U_arr = np.array((df.attrs["U_m_s"]**2 + df.attrs["V_m_s"]**2)**0.5)
        B_arr = np.array(gdf[gdf["bkg"]]["xch4_ppb"])

        for i in range(len(E_means)):
            U = np.mean(np.random.choice(U_arr, (len(U_arr),)))*60*60 # [m/h]
            B = np.mean(np.random.choice(B_arr, (len(B_arr),))) # [ppb]
            gdf["delta_omega"] = ((gdf["xch4_ppb"]-B)*1e-9*
                                  gdf["dry_air_mol_m2"]*16.04*1e-3) # [kg/m2]
            gdf["delta_omega_times_height"] = (gdf["delta_omega"]*
                                               gdf["height_m"]) # [kg/m]
            sums = (gdf[gdf["plume"]].groupby("lon_center")
                    [["delta_omega_times_height"]].sum()) # [kg/m]
            T_arr = np.array(sums.values.flatten())
            T = np.mean(np.random.choice(T_array, (len(T_array),))) # [kg/m]
            E_i[i] = U*T*24*365*1e-6 # [kt/yr]

        E_min = np.percentile(E_i, 2.5)
        E_max = np.percentile(E_i, 97.5)

    # Clean up files
    [os.remove(f"{idx+1}.pkl") for idx in range(0,4)]
    [os.remove(f) for f in glob.glob("slurm*.out")]

    return fig, E_means, E_min, E_max