#!/bin/bash

sbatch --output=/scratch/$USER/logs/slurm-%j-%x.out --error=/scratch/$USER/logs/slurm-%j-%x.error sbatch/earthquake.sh ~/scratch/code-snapshots/earthquake