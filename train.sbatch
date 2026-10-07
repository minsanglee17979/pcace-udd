#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --job-name=ar-udd-training
#SBATCH --output=train-%A_%a.out

set -e
cd "$SLURM_SUBMIT_DIR"
seeds=(101 202 303 404)
bash "training/bb-${seeds[$SLURM_ARRAY_TASK_ID]}/run.sh"
