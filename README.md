## Introduction
This repository contains code for Balasus et al. (2024).

## Creating the environment
I use micromamba as a package manager. You can install it [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Then, from the command line, run:
```
micromamba create -f environment.yml
```

## Downloading the data
The data needed are (1) blended TROPOMI+GOSAT and (2) HRRR data (winds, temperature, surface pressure, precipitation). First, go to `config.json` and specify the directories where each of these datasets can be stored. Then, run the corresponding Python script for each in `scripts/`. I use SLURM to allocate resources for each job:
```
sbatch -J blended -p sapphire -t 0-24:00 --mem=1000000 -c 112 \
       --wrap "source ~/.bashrc; micromamba activate ldf_env; \
               python -B -m scripts.download-blended"

months=()
for i in {0..60}; do months+=( $(date -d "20190101+${i} month" +%Y-%m-%d) ); done
for i in {0..59}; do
sbatch -J hrrr -p sapphire -t 0-24:00 --mem=500000 -c 48 \
        --wrap "source ~/.bashrc; micromamba activate ldf_env; \
        python -B -m scripts.download-hrrr ${months[i]} ${months[i+1]}"
done
```

## Running the analysis
The file `config.json` lists the four landfills along with their coordinates and GHGRP IDs. The Python script `assemble.py` will oversample TROPOMI data for 2019-2023 for each of the landfills, apply a cross sectional flux analysis to infer emissions from the resulting plumes, and scrape relevant data from the EPA FLIGHT tool.
```
sbatch -J assemble -p sapphire -t 0-06:00 --mem=64000 -c 4 \
        --wrap "source ~/.bashrc; micromamba activate ldf_env; \
        python -B -m scripts.assemble"
```