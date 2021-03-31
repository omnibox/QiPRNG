#!/bin/bash
#
#SBATCH --job-name=statistical_tests
#SBATCH --output=slurm_out.txt
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -p short-40core
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=aaron.f.gregory@stonybrook.edu

module load anaconda/3

cd /gpfs/home/afgregory/QiPRNG/src

python DataProcessing.py