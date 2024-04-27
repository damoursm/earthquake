# Local development
## Requirements
1. Download the data sample from: http://repo.pi.ingv.it/instance/Instance_sample_dataset_v2.tar.bz2
1. Create an environment variable file, called `.env` at the root of the project. This is to set the `data` and `output` paths. Change the variable value to your actual data and output location. The folder and files should exist before you run the code.
  - EVENT_HDF5_FILE="data/instance_samples/Instance_events_counts_10k.hdf5"
  - EVENT_METADATA_FILE="data/instance_samples/metadata_Instance_events_10k.csv"
  - NOISE_HDF5_FILE="data/instance_samples/Instance_noise_1k.hdf5"
  - NOISE_METADATA_FILE="data/instance_samples/metadata_Instance_noise_1k.csv"
  - FINAL_OUTPUT_DIR="output"
  - TEMP_DIR="temp"
1. Create a `venv` environment using **python 3.11** and install the packages in the `requirements.txt` found at the root of the project. Use this environment to run the code.

## Training existing models (in the report)
### EQ Model, based on transformers:
1. Run `python train_my_eq.py` from the root of the project
2. Check progress in the terminal. At the end, the result will be added to the `output` file mentionned in the `.env` file.
### CNN model
1. Run `python train_my_cnn1.py` or `python train_my_cnn2.py`
2. Check progress in the terminal. At the end, the result will be added to the `output` file mentionned in the `.env` file.

## Train your own model
1. Copy one of the file `train_my_eq.py` and edit it to change the model and hyperparameters
2. Train your model by raining `python train_my_own_model.py` 


# Cluster training

## Data on the cluster
Instance data have been downloaded and available here: `~/projects/def-sponsor00/earthquake/data/instance`\
STEAD data are not yet downloaded but can be added here: `~/projects/def-sponsor00/earthquake/data/stead`\
You want to downlaod in parallel for STEAD:
1. put all your the urls in a file (files.txt)
1. and do:  `cat files.txt | xargs -n 1 -P 0 wget -q`\
   -P 0 let xargs choose the number of parallels work. You can assign a hard number if you want

## Must
1. Use `tmux` to run your session. Once connected on the server:
  - Type `tmux` to start a new session or `tmux attach` to recover from an old session. I suggest using it as tmux will keep your terminal session running even if you loose connection. Otherwise you might need to start from scratch
  - Type `ctrl-b + %` to split your screen and `ctrl-b <arrow>` to navigate through the panes. I use it to be able to run simultaneously multiple terminals as one might be blocked by a long running task.
  - Use `ctrl-b + z` to toggle one pane full screen or not
  - Type `exit` to close tmux pane

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
1. Keep in the home folder and start training. Replace `train_transformer_elisee.py` with the file containing the code (For EqModel, use `train_my_eq.py` and for Cnn, use `train_my_cnn.py`)
     - `cd ~`
     - `./sbatch/run.sh -p train_transformer_elisee.py`. Here you can optionally specifiy few arguments. `-m 16Gb` for memory (by default `8Gb`). `-t hh:mm:ss` for how long to run (by default 1H). `-p /train_xxx.py` for the file to execute (by default it will run train.py). `-c 1` for the number of cpu to use. `g 1` for the number of gpu to use.
1. Once training is done, the files will be in the `FINAL_OUTPUT_DIR` specified in the `.env`. To download them 1 by 1 on your local computer, use this command line: `scp <username>@ift6759.calculquebec.cloud:/scratch/<username>/output/default-train/<filename> <local path e.g /Users/ekabore/Downloads>`
1. Useful slurm commands:
     - squeue -u username : will show the current job being submitted
     - scontrol show job jobid : show details about the job
     - scancel jobid : cancels a job
