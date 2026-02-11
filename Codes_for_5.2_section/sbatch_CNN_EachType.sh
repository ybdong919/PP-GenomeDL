#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=eachtype_fhe_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300gb
#SBATCH --time=336:00:00
#SBATCH --output=eachtype.%j.out
#SBATCH --error=eachtype.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END



#python FHE_CNN_cancer3.py
#python FHE_CNN_cancer2.py
#python FHE_CNN_cancer2-2.py

#python test_k-fold.py
#python FHE_CNN2d_cancer.py

#python test_eachtype.py

python test_filtertype.py