# earthquake

## Steps to train in the cluster
1. Using the terminal, login to your cluster, preferrably through ssh (e.g: ssh username@ift6759.calculquebec.cloud)
1. Create a folder where you will clone your repo: mkdir documents
1. Get in the folder and clone the repo or pull if already cloned (using ssh preferrably):
     - cd documents
     - git clone git@github.com:damoursm/earthquake.git OR git pull
1. Get in the scripts folder and run the setup script. That will create sbatch and scratch folder and move code and scripts to proper location
     - cd scripts
     - ./setup.sh
1. Go back to home and execute the run to start training
     - cd ~
     - ./sbatch/run.sh
1. You can customize the sbatch parameters in run.sh file
1. Useful slurm commands:
     - squeue -u username : will show the current job being submitted
     - scontrol show job jobid : show details about the job
     - scancel jobid : cancels a job
