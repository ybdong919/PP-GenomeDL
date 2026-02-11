#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=xmpfhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400gb
#SBATCH --time=336:00:00
#SBATCH --output=mpfhe.%j.out
#SBATCH --error=mpfhe.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END




# Test with FHE
python fl_fhe_v16_nan_fixed_FINAL_31.py --client 3 --hidden_size 512 --epochs 20 --device cpu --batch_size 8 --lr 0.0001 --expname FHE_FIXED --train_data train_data_norm_log10 --test_data test_data_norm_log10

