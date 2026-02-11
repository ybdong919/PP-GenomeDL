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


# Table 1: Compression methods comparison
python fl_fhe_v17_gradient_compression_1.py --client 2 --hidden_size 512 --epochs 10 --device cpu --batch_size 8 --lr 0.0001 --expname pub_topk_10 --train_data train_data_norm_log10 --test_data test_data_norm_log10 --compression top_k --compress_ratio 0.1
#python fl_fhe_v17_gradient_compression_1.py --no_compression --epochs 10 --expname pub_baseline --client 2 --hidden_size 512 --device cpu --batch_size 8 --lr 0.0001 --train_data train_data_norm_log10 --test_data test_data_norm_log10
python fl_fhe_v17_gradient_compression_1.py --compression top_k --compress_ratio 0.5 --epochs 10 --expname pub_topk_50 --client 2 --hidden_size 512 --device cpu --batch_size 8 --lr 0.0001 --train_data train_data_norm_log10 --test_data test_data_norm_log10
#python fl_fhe_v17_gradient_compression_1.py --compression top_k --compress_ratio 0.1 --epochs 10 --expname pub_topk_10 --client 2 --hidden_size 512 --device cpu --batch_size 8 --lr 0.0001 --train_data train_data_norm_log10 --test_data test_data_norm_log10
python fl_fhe_v17_gradient_compression_1.py --compression top_k --compress_ratio 0.05 --epochs 10 --expname pub_topk_5 --client 2 --hidden_size 512 --device cpu --batch_size 8 --lr 0.0001 --train_data train_data_norm_log10 --test_data test_data_norm_log10
python fl_fhe_v17_gradient_compression_1.py --compression top_k --compress_ratio 0.01 --epochs 10 --expname pub_topk_1 --client 2 --hidden_size 512 --device cpu --batch_size 8 --lr 0.0001 --train_data train_data_norm_log10 --test_data test_data_norm_log10

