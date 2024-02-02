#!/bin/bash
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80000
#SBATCH --time=48:00:00
#SBATCH --job-name=deepcdr
#SBATCH --output=reg_normal_final.out
#SBATCH --error=reg_normal_final.err
#SBATCH --cpus-per-gpu=6
#SBATCH --exclude=gpu01.exbio.wzw.tum.de

python /nfs/home/students/e.albrecht/DeepCDR_latest_CUDA/Adapted_DeepCDR_Code/prog/run_DeepCDR.py -gpu_id 0
