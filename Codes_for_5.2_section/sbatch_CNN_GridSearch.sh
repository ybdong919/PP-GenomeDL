#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=grid_fhe_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300gb
#SBATCH --time=120:00:00
#SBATCH --output=grid_fhe_cnn.%j.out
#SBATCH --error=grid_fhe_cnn.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=hpg-b200
#SBATCH --gpus=2


#python FHE_CNN_cancer3.py
#python FHE_CNN_cancer2.py
#python FHE_CNN_cancer2-2.py
#python test.py
#python test_k-fold.py
#python FHE_CNN2d_cancer.py
python test_GridSearch.py