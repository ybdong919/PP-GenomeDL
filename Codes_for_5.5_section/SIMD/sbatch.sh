#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=op-mpfhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400gb
#SBATCH --time=336:00:00
#SBATCH --output=mpfhe.%j.out
#SBATCH --error=mpfhe.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END



python fl_fhe_v16_simd_fixed.py --client 2 --hidden_size 512 --epochs 10 --device cpu --batch_size 8 --lr 0.0001 --expname v16_simd_fixed --train_data train_data_norm_log10 --test_data test_data_norm_log10

