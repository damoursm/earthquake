#!/bin/bash
#SBATCH --job-name=earthquake                                               # Job name
#SBATCH --cpus-per-task=1                                                   # Ask for 1 CPUs
#SBATCH --mem=1Gb                                                           # Ask for 1 GB of RAM
#SBATCH --output=/scratch/logs/slurm-%j-%x.out                              # log file
#SBATCH --error=/scratch/logs/slurm-%j-%x.error                             # log file
#SBATCH --time=00:01:00                                                     # Run time

# Arguments
# $1: Path to code directory

# Copy code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git" --exclude "scripts" --exclude "env" --exclude "data"

cd $SLURM_TMPDIR/"$1"

# Setup environment
module purge
module load StdEnv/2020
module load python/3.11
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py

