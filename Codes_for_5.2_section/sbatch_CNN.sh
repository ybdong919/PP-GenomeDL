#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=fhe_cnn2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=800gb
#SBATCH --time=72:00:00
#SBATCH --output=fhe_cnn2d.%j.out
#SBATCH --error=fhe_cnn2d.err
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

#python FHE_CNN2d_29cancer.py
#python FHE_CNN2d_29cancer_Noearlystop.py

python FHE_CNN2d_cancer_aftertests.py