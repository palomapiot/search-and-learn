# IRLab Usage Guide

This guide provides instructions on building a Singularity image and running experiments as a Slurm job.

## Building the Singularity Image
To build the Singularity image, run the following command:
```bash
sudo singularity build --disable-cache /mnt/experiments/slurm/singularity-containers/search-and-learn.sif singularity/image.def
```
This command creates a Singularity container using the definition file located at `singularity/image.def`.

## Running Experiments with Slurm
To run experiments as a Slurm job, execute:
```bash
sbatch src/scripts/demo.sh
```
This submits the `demo.sh` script as a job to the Slurm workload manager.

## Additional Information
For more details about this project, refer to the original [README](README_original.md).