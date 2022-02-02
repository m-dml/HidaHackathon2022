#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_forecast_conda
#SBATCH --partition=p2GPU40
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_forecast_conda.txt
#SBATCH --cpus-per-task=10

export CUDA_CACHE_DISABLE=1

data_dir=/gpfs/work/machnitz
weights_path=/gpfs/work/machnitz/weights/

/gpfs/home/machnitz/miniconda3/envs/hydra/bin/python /gpfs/home/machnitz/HidaHackathon2022/training.py --save_dir "$PWD" --data_dir ${data_dir} --weights_path ${weights_path}
