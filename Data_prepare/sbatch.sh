#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=ppml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400gb
#SBATCH --time=336:00:00
#SBATCH --output=ppml.%j.out
#SBATCH --error=ppml.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END

#Step1: process data
python ./01.processData.py

#Step2: divide data to train (80%) and test data (20%)
python ./01.processData_divideData.py

