import os
import json
import subprocess
with open("config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":

    # Oversample around each landfill for each year
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

            if os.path.exists(f"{name}.pkl"):
                continue
            
            # Submit the job using SLURM, waiting if this is the last job
            command = [
                "sbatch", "-p", "sapphire,huce_ice", "-t", "180",
                "--mem", "128000", "-c", "64", "--wrap", (f"source ~/.bashrc; "
                f"micromamba activate ldf_env; "
                f"python scripts/perform-oversampling.py "
                f"{start_time} {end_time} {lon_min} {lon_max} {lon_res} {lat_min} "
                f"{lat_max} {lat_res} {name} {wind_rotate} "
                f"{source_lon} {source_lat}")
            ]
            if count == 20:
                command.insert(1, "--wait")
            subprocess.run(command)
            count += 1