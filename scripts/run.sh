#!/bin/bash

#Add the following parameters to override the default values if needed
# --job-name=name_of_the_job       
# --cpus-per-task=1    

# The last 2 arguments are the sbatch script and the path to the code                                                       

sbatch --mem=1Gb --time:00:05:00 --output=/scratch/$USER/logs/slurm-%j-%x.out --error=/scratch/$USER/logs/slurm-%j-%x.error ~/sbatch/earthquake.sh ~/scratch/code-snapshots/earthquake