#!/bin/bash

#PBS -N smp-light-evasion-v6-ablations-sdsTrue_kltNone_bs64000
#PBS -l nodes=1:ppn=all
#PBS -l walltime=72:00:00
#PBS -l mem=26.9gb
#PBS -m abe

cd $PBS_O_WORKDIR

# Setup anaconda environment
conda init
source ~/.bashrc
conda activate thesis

export PYTHONPATH="thesis_manta_ray:$HOME"

# Update code
cd thesis_manta_ray
git pull --recurse-submodules

# Setup wandb
wandb_api_key=`cat $HOME/wandb_key.txt`
export WANDB_API_KEY=$wandb_api_key

# Run
OMP_PROC_BIND="" OMP_NUM_THREADS=1 python -m "evolution_simulation"