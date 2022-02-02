#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_evaluation_conda
#SBATCH --partition=haicore-gpu4
#SBATCH --reservation=ai_hero
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_eval_conda.txt

export CUDA_CACHE_DISABLE=1

group_name=E1
group_workspace=/hkfs/work/workspace/scratch/bh6321-${group_name}

data_dir=/hkfs/work/workspace/scratch/bh6321-energy_challenge/data
forecast_path=${group_workspace}/AI-HERO-Energy/forecasts.csv


/home/haicore-project-hereon/eu7630/miniconda3/envs/plankton/bin/python ${group_workspace}/AI-HERO-Energy/evaluation.py --save_dir "$PWD" --data_dir ${data_dir} --forecast_path ${forecast_path}
