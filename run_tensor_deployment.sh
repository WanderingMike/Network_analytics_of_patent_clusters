#!/bin/bash

#SBATCH --job-name=tensor_deployment
#SBATCH --output=std_out/%x-%j.out
#SBATCH --error=std_out/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:0
#SBATCH --chdir=/cluster/raid/home/michael.tsesmelis/SideProject

echo $(pwd)

#print gpu configuration for this job
nvidia-smi

#Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES

source ./bin/activate

python tensor_deployment.py

deactivate
