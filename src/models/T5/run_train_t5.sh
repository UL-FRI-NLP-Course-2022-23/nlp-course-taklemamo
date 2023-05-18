#!/bin/bash
#SBATCH --job-name=train_t5
#SBATCH --output=train.out
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --reservation=fri-vr
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBARCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G

srun python ./train_t5.py