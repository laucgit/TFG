#!/bin/bash
#SBATCH -J dome
#SBATCH -o dome_%j.out
#SBATCH -e dome_%j.err
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH --mem=4G

module load julia

julia scripts/sweep_experiments.jl