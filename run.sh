#!/bin/env bash
#SBATCH --partition=lln
#SBATCH --time=200:00:00
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
source /home/0477699/.bashrc    
conda activate pmp_test
srun python3 meep_code.py $1