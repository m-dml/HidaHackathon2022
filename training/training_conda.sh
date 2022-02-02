#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_forecast_conda
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_forecast_conda.txt
#SBATCH --cpus-per-task=30

export CUDA_CACHE_DISABLE=1

group_name=E1
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}

data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data
save_dir=/hkfs/work/workspace/scratch/bh6321-E1/weights

/home/haicore-project-hereon/eu7630/miniconda3/envs/plankton/bin/python ${group_workspace}/HidaHackathon2022/training.py --save_dir ${save_dir} --data_dir ${data_dir}
