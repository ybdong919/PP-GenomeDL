#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=ppml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
#SBATCH --time=336:00:00
#SBATCH --output=ppml_pred.%j.out
#SBATCH --error=ppml_pred.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END

python ./scripts/class_pred.py