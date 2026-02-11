#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=attack
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
#SBATCH --time=336:00:00
#SBATCH --output=attack.%j.out
#SBATCH --error=attack.err
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=hpg-b200
#SBATCH --gpus=2

#Model Inversion Attack

python attack_CNN_2.py --model model/CNN_no_DP_no_FHE_modelbest.tar
python attack_CNN_2.py --model model/CNN_DP_no_FHE_modelbest.tar
python attack_CNN_2.py --model model/CNN_DPe10_no_FHE_modelbest.tar
python attack_CNN_2.py --model model/CNN_DPe10_no_FHE_bigdata_modelbest.tar
python attack_CNN_2.py --model model/CNN_no_DP_no_FHE_bigdata_modelbest.tar


python attack_CNN_no_Quantization.py --model model/CNN2d_no_Quantization_smalldata_modelbest.tar


python attack_CNN_no_Quantization_37cancers.py --model model/CNN2d_no_Quantization_smalldata_37cancers_modelbest.tar
python attack_CNN_no_Quantization_37cancers.py --model model/CNN2d_no_Quantization_largedata_37cancers_modelbest.tar

python attack_CNN_no_Quantization_37cancers_X1.py --model model/CNN2d_no_Quantization_largedata_37cancers_X1_modelbest.tar
python attack_CNN_no_Quantization_37cancers_X2.py --model model/CNN2d_no_Quantization_largedata_37cancers_X2_modelbest.tar