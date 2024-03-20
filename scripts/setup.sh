#!/bin/bash

mkdir ~/sbatch
mkdir ~/scratch
mkdir ~/scratch/logs
mkdir ~/scratch/code-snapshots
mkdir ~/scratch/output
mkdir ~/scratch/output/default-train

rsync -av ../../earthquake ~/scratch/code-snapshots/ --exclude .git --exclude scripts --exclude env --exclude data

rsync -av ./earthquake.sh ~/sbatch/
rsync -av ./run.sh ~/sbatch/
