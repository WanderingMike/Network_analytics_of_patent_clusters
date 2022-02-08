#!/bin/bash

#SBATCH --job-name=managerial_layer
#SBATCH --output=std_out/main/%x-%j.out
#SBATCH --error=std_out/main/%x-%j.err
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/michael.tsesmelis/SideProject

echo $(pwd)

#print gpu configuration for this job
#nvidia-smi

#Verify gpu allocation (should be 1 GPU)
#echo $CUDA_VISIBLE_DEVICES

source ./bin/activate

echo "*** Starting jupyter on " $(hostname)
jupyter notebook --no-browser --port=9999


deactivate
