#!/bin/bash
#SBATCH --job-name="search-and-learn"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=40G
#SBATCH -o /mnt/gpu-fastdata/paloma/search-and-learn/logs/%x-%j.out
#SBATCH -e /mnt/gpu-fastdata/paloma/search-and-learn/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/search-and-learn.sif"

export HF_TOKEN=HF_TOKEN

singularity run --disable-cache --pwd $(pwd) --nv --bind /mnt:/mnt $SIF python3 src/test_time_compute.py recipes/hate-speech/best_of_n.yaml