#!/bin/bash

#SBATCH -o %x_output_%j.txt
#SBATCH -e %x_error_%j.txt
#SBATCH --job-name=pywgcna
#SBATCH --mem=128000mb
#SBATCH --cpus-per-task=60
#SBATCH --time=48:00:00

python3 /cmnfs/home/students/a.schmucklermann/pywgcna.py
