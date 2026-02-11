#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=ppml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=700gb
#SBATCH --time=336:00:00
#SBATCH --output=ppml.%j.out
#SBATCH --error=ppml.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END

python ./scripts/test_2.py

