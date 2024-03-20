#!/bin/bash


while getopts m:t:p:c:g flag
do
    case "${flag}" in
        m) memory=${OPTARG};;
        t) time=${OPTARG};;
        p) python_file=${OPTARG};;
        c) cpu=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

if [ -z "$memory" ]
then
    memory=8Gb
fi

if [ -z "$python_file" ]
then
    python_file=train.py
fi

if [ -z "$cpu" ]
then
    cpu=1
fi

if [ -z "$gpu" ]
then
    gpu=1
fi

output_file=/scratch/$USER/logs/slurm-%j-%x.out
error_file=/scratch/$USER/logs/slurm-%j-%x.error
sbatch_executable_file=~/sbatch/earthquake.sh
code_folder=~/scratch/code-snapshots/earthquake
data_folder=~/projects/def-sponsor00/earthquake/data/instance


if [ -z "$time" ]
then
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --gpus-per-node=$gpu  --cpus-per-task=$cpu --mem=$memory --output=$output_file --error=$error_file $sbatch_executable_file $code_folder $data_folder $python_file
    echo "Scheduled $python_file to run with $memory memory. See output in $output_file"
else
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --time=$time --gpus-per-node=$gpu  --cpus-per-task=$cpu --mem=$memory --output=$output_file --error=$error_file $sbatch_executable_file $code_folder $data_folder $python_file
    echo "Scheduled $python_file to run with $memory memory for $time long. See output in $output_file"
fi