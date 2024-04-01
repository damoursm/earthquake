#!/bin/bash
#SBATCH --job-name=earthquake                                               # Job name
#SBATCH --cpus-per-task=1                                                   # Ask for 1 CPUs
#SBATCH --mem=1Gb                                                           # Ask for 1 GB of RAM

# Arguments
# $1: Path to code directory

# Copy code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git" --exclude "scripts" --exclude "env" --exclude "data"

# Copy data to the SLURM_TMPDIR to increase speed
rsync -av --relative "$2" $SLURM_TMPDIR --exclude "Instance_events_counts.hdf5" --exclude "Instance_noise.hdf5" --exclude "metadata_Instance_events_v2.csv" --exclude "metadata_Instance_noise.csv"

export SLURM_TMPDIR_EVENT_HDF5_FILE=$SLURM_TMPDIR/"$2"/earthquake_20k.hdf5
export SLURM_TMPDIR_EVENT_METADATA_FILE=$SLURM_TMPDIR/"$2"/earthquake_20k.csv
export SLURM_TMPDIR_NOISE_HDF5_FILE=$SLURM_TMPDIR/"$2"/noise_20k.hdf5
export SLURM_TMPDIR_NOISE_METADATA_FILE=$SLURM_TMPDIR/"$2"/noise_20k.csv

# export SLURM_TMPDIR_EVENT_HDF5_FILE=/project/def-sponsor00/earthquake/data/instance/Instance_events_counts.hdf5
# export SLURM_TMPDIR_EVENT_METADATA_FILE=/project/def-sponsor00/earthquake/data/instance/metadata_Instance_events_v2.csv
# export SLURM_TMPDIR_NOISE_HDF5_FILE=/project/def-sponsor00/earthquake/data/instance/Instance_noise.hdf5
# export SLURM_TMPDIR_NOISE_METADATA_FILE=/project/def-sponsor00/earthquake/data/instance/metadata_Instance_noise.csv

cd $SLURM_TMPDIR/"$1"

ls -a

# Setup environment
module purge
module load StdEnv/2020
module load python/3.11
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt

python $3

