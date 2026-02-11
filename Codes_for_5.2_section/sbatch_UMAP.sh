#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=UMAP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200gb
#SBATCH --time=5:00:00
#SBATCH --output=UMAP.out
#SBATCH --error=UMAP.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END




#python FHE_CNN_cancer3.py
#python FHE_CNN_cancer2.py
#python FHE_CNN_cancer2-2.py

#python test_k-fold.py
#python FHE_CNN2d_cancer.py

#python test_CNN_BestHyperparameters.py
#python test_CNN_BestHyper_EarlyStop.py
#python test_CNN_BestHyper_EarlyStop_30types.py
#python test_CNN_BestHyper_EarlyStop_34types.py

python UMAP.py