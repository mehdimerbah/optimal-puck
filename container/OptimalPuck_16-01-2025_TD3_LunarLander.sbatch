#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=OptimalPuck_16-01-2025_TD3_LunarLander

#resources:

#SBATCH --cpus-per-task=8

#SBATCH --partition=day

#SBATCH --mem-per-cpu=6G

#SBATCH --gres=gpu:2080ti:1

#SBATCH --time=12:00:00

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=MAILUSERNAME@uni-tuebingen.de



####
#b) Execute training
####

log_message "Starting job at $(date)"
log_message "Running on host: $(hostname)"
log_message "Working directory: $(pwd)"
log_message "\nStarting training..."
singularity exec --nv optimal-puck.sif python3 scripts/train_model.py \
    --experiment-id lunarlander-v3_TD3_20250116_1218 \
    --model-type td3 2>&1 | tee -a "$LOG_FILE"

log_message "\nJob finished at $(date)"
log_message "DONE!"
