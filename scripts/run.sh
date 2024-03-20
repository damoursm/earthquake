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
    memory=8Gb
fi

if [ -z "$python_file" ]
then
    python_file=train.py
fi

output_file=/scratch/$USER/logs/slurm-%j-%x.out
error_file=/scratch/$USER/logs/slurm-%j-%x.error
sbatch_executable_file=~/sbatch/earthquake.sh
code_folder=~/scratch/code-snapshots/earthquake
data_folder=~/projects/def-sponsor00/earthquake/data/instance


if [ -z "$time" ]
then
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --mem=$memory --output=$output_file --error=$error_file $sbatch_executable_file $code_folder $data_folder $python_file
    echo "Scheduled $python_file to run with $memory memory. See output in $output_file"
else
    # The last 3 arguments are the sbatch script and the path to the code and python file to execute 
    sbatch --mem=$memory --time=$time --output=$output_file --error=$error_file $sbatch_executable_file $code_folder $data_folder $python_file
    echo "Scheduled $python_file to run with $memory memory for $time long. See output in $output_file"
fi