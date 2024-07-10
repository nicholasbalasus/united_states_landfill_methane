## Introduction
This repository contains code for Balasus et al. (2024).

## Creating Environment
I use micromamba as a package manager. You can install it [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Then, from the command line, run:
```
micromamba create -f environment.yml
```

## EPA FLIGHT Data
Emissions report from individual landfills are scraped from the EPA [FLIGHT](https://ghgdata.epa.gov/ghgp/main.do?site_preference=normal) tool. The function to do this is in `scripts/scraper.py`. Running `notebooks/flight.ipynb` scrapes the data for all landfills and plots the aggregated data.

## TROPOMI Emissions Estimates
The data needed are (1) blended TROPOMI+GOSAT and (2) HRRR data (winds, temperature, surface pressure, precipitation). First, go to `config.json` and specify the directories where each of these datasets can be stored. Then, run the corresponding Python script for each in `scripts/`. I use SLURM to allocate resources for each job:
```
# Blended TROPOMI+GOSAT
sbatch -J blended -p sapphire -t 0-24:00 --mem=1000000 -c 112 \
       --wrap "source ~/.bashrc; micromamba activate ldf_env; \
               python -B -m scripts.download-blended"

# HRRR Data
months=()
for i in {0..60}; do months+=( $(date -d "20190101+${i} month" +%Y-%m-%d) ); done
for i in {0..59}; do
sbatch -J hrrr -p sapphire -t 0-24:00 --mem=500000 -c 48 \
        --wrap "source ~/.bashrc; micromamba activate ldf_env; \
        python -B -m scripts.download-hrrr ${months[i]} ${months[i+1]}"
done
```