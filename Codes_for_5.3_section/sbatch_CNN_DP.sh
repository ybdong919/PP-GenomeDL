#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=fhe_dp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400gb
#SBATCH --time=72:00:00
#SBATCH --output=fhe_dp.%j.out
#SBATCH --error=fhe_dp.err
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

#python FHE_CNN2d_cancer_aftertests.py

#### final script
python FHE_CNN2d_DP.py

### output best no-FHE-compiled-model for Attack test, and FHE-compile model with DP for client/server deployment
python FHE_CNN2d_DP_output_model.py

### output best no-DP-no-FHE-compiled-model as control for Attack test, and FHE-compile model without DP for client/server deployment
python FHE_CNN2d_DP_output_model_exclude_DP.py

### Based on a large sample data, output best no-FHE-compiled-model for Attack test, and FHE-compile model with DP for client/server deployment
python FHE_CNN2d_DP_output_model_input_large_data.py
python FHE_CNN2d_DP_output_model_input_large_data_exclude_DP.py

### output best non-quantization CNN2d model, based on small sample and 29 cancer types. Note: No FHE in the code, because FHE requires quantizated model. 
python CNN2d_no_Quantization_output_model_small_data.py

### output best non-quantization CNN2d model, based on small sample and 37 cancer types. Note: No FHE in the code, because FHE requires quantizated model. 
python CNN2d_no_Quantization_output_model_small_data_37cancers.py

### output best non-quantization CNN2d model, based on large sample and 37 cancer types.
#python CNN2d_no_Quantization_output_model_large_data_37cancers.py
python CNN2d_no_Quantization_output_model_large_data_37cancers-X1.py
python CNN2d_no_Quantization_output_model_large_data_37cancers-X2.py