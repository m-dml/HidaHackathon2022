#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_forecast_conda
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_forecast_conda.txt
#SBATCH --cpus-per-task=72

export CUDA_CACHE_DISABLE=1
module load devel/python/3.8

group_name=E1
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}

data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data
weights_path=${group_workspace}/weights/

python -u ${group_workspace}/Dynamic-Ants/forecast.py --save_dir "$PWD" --data_dir ${data_dir} --weights_path ${weights_path}
