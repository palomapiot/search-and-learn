#!/bin/bash
#SBATCH --job-name="search-and-learn"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=10G
#SBATCH -o /mnt/gpu-fastdata/eliseo/search-and-learn/logs/%x-%j.out
#SBATCH -e /mnt/gpu-fastdata/eliseo/search-and-learn/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/search-and-learn.sif"

export HF_TOKEN=YOUR_ACCESS_TOKEN

singularity run --disable-cache --pwd $(pwd) --nv --bind /mnt:/mnt $SIF python3 src/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml