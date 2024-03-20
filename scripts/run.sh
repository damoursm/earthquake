#!/bin/bash


while getopts m:t:p: flag
do
    case "${flag}" in
        m) memory=${OPTARG};;
        t) time=${OPTARG};;
        p) python_file=${OPTARG};;
    esac
done

if [ -z "$memory" ]
then
    memory=32Gb
fi

if [ -z "$python_file" ]
then
    python_file=train.py
fi


if [ -z "$time" ]
then
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --mem=$memory --output=/scratch/$USER/logs/slurm-%j-%x.out --error=/scratch/$USER/logs/slurm-%j-%x.error ~/sbatch/earthquake.sh ~/scratch/code-snapshots/earthquake $python_file
    echo "Scheduled $python_file to run with $memory memory"
else
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --mem=$memory --time=$time --output=/scratch/$USER/logs/slurm-%j-%x.out --error=/scratch/$USER/logs/slurm-%j-%x.error ~/sbatch/earthquake.sh ~/scratch/code-snapshots/earthquake $python_file
    echo "Scheduled $python_file to run with $memory memory for $time long"
fi