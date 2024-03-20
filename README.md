# earthquake

## Data on the cluster
Instance data have been downloaded and available here: `~/projects/def-sponsor00/earthquake/data/instance`\
STEAD data are not yet downloaded but can be added here: `~/projects/def-sponsor00/earthquake/data/stead`\
You want to downlaod in parallel for STEAD:
1. put all your the urls in a file (files.txt)
1. and do:  `cat files.txt | xargs -n 1 -P 0 wget -q`\
   -P 0 let xargs choose the number of parallels work. You can assign a hard number if you want

## What you need
1. An environment variable, called `.env` at the root of the project. This is set the data and output paths. Change the variable value to actual location. The folder and files should exit.
  - EVENT_HDF5_FILE="data/instance_samples/Instance_events_counts_10k.hdf5"
  - EVENT_METADATA_FILE="data/instance_samples/metadata_Instance_events_10k.csv"
  - NOISE_HDF5_FILE="data/instance_samples/Instance_noise_1k.hdf5"
  - NOISE_METADATA_FILE="data/instance_samples/metadata_Instance_noise_1k.csv"
  - FINAL_OUTPUT_DIR="output"
  - TEMP_DIR="temp"
1. Duplicate the `train.py` file at the root of the project and edit it to train your own model following the structure. For instance, I can create `train_transformer_elisee.py`.

## Steps to train on your computer
1. Make sure your environment file (`.env`) is properly set
1. Execute your train file. For instance `python3 train_transformer_elisee.py` in terminal or using debugger in your IDE.

## Steps to train in the cluster
1. Using the terminal, login to your cluster, preferrably through ssh (e.g: ssh username@ift6759.calculquebec.cloud)
1. Create a folder where you will clone your repo: `mkdir documents`
1. Get into the folder and clone the repo or pull if already cloned before (using ssh preferrably):
     - `cd documents`
     - `git clone git@github.com:damoursm/earthquake.git` OR `git pull`
1. Get into the scripts folder and run the setup script. That will create sbatch and scratch folder and move code and scripts to proper location
     - `cd scripts`
     - `./setup.sh`
1. Go back to home Add your `.env` file in the code folder
     - `cd ~`
     - `vim scratch/code-snapshots/earthquake/.env`
     - Set the variable as in the above section. You can leave all variables except `FINAL_OUTPUT_DIR` with empty values (`""`) as code is detecting cluster and selecting where the code is.
     - Set `FINAL_OUTPUT_DIR="scratch/<your username>/output/default-train"`. `default-train` is used as default but you can change it if you want to save output of different experiments. Just make sure the folder exists
1. Keep in the home folder and start training
     - `cd ~`
     - `./sbatch/run.sh -p train_transformer_elisee.py`. Here you can optionally specifiy few arguments. `-m 16Gb` for memory (by default `8Gb`). `-t hh:mm:ss` for how long to run (by default 1H). `-p /train_xxx.py` for the file to execute (by default it will run train.py). `-c 1` for the number of cpu to use. `g 1` for the number of gpu to use.
1. Once training is done, the files will be in the `FINAL_OUTPUT_DIR` specified in the `.env`. To download them 1 by 1 on your local computer, use this command line: `scp <username>@ift6759.calculquebec.cloud:/scratch/<username>/output/default-train/<filename> <local path e.g /Users/ekabore/Downloads>`
1. Useful slurm commands:
     - squeue -u username : will show the current job being submitted
     - scontrol show job jobid : show details about the job
     - scancel jobid : cancels a job
