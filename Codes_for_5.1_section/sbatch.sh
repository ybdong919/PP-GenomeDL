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
#python ./01.processData.py

#Step2: divide data to train (80%) and test data (20%)
#python ./01.processData_divideData.py

#Step3: centrally trained method (#client=1)
python 02.simulationApp_debug_cpu.py --device cpu --mode SGD --client 1 --epochs 200 --batch_size 8 --lr 0.001 --expname PureSGD0 --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_GTEM_1head_1attention.py --device cpu --mode SGD --client 1 --epochs 10 --batch_size 8 --lr 0.001 --expname TransformerPureSGD0 --train_data train_data_norm_log10 --test_data test_data_norm_log10

#Step4: FL trained method (#client=5)
python 02.simulationApp_debug_cpu.py --device cpu --mode SGD --client 5 --epochs 200 --batch_size 8 --lr 0.001 --expname FL --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_GTEM_1head_1attention.py --device cpu --mode SGD --client 5 --epochs 10 --batch_size 8 --lr 0.001 --expname TransformerFL --train_data train_data_norm_log10 --test_data test_data_norm_log10

#Step5: DP trained method with/without FL
for e in 400 300 200 150 100 50 10 1
do
python 02.simulationApp_debug_cpu.py --device cpu --mode DP --client 1 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname PureSGD0_DP_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-corr-noise.py --device cpu --mode DP --client 1 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname PureSGD0_DP_corrd10000_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-gm-noise.py --device cpu --mode DP --client 1 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname PureSGD0_DP_gm_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-laplace-noise.py --device cpu --mode DP --client 1 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname PureSGD0_DP_lap_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-laplace-noise.py --device cpu --mode DP --client 10 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname FL10_DP_lap_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-laplace-noise.py --device cpu --mode DP --client 20 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname FL20_DP_lap_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-laplace-noise.py --device cpu --mode DP --client 40 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname FL40_DP_lap_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_DP-stable-noise.py --device cpu --mode DP --client 1 --epochs 200 --batch_size 8 --lr 0.001 --epsilon ${e} --expname PureSGD0_DP_stable_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
python 02.simulationApp_debug_cpu_GTEM_1head_1attention.py --device cpu --mode DP --client 1 --epochs 10 --batch_size 8 --lr 0.001 --epsilon ${e} --expname DP_e${e} --train_data train_data_norm_log10 --test_data test_data_norm_log10
done

#Step6: Transformer model with DP and FL
python 02.cpu_GTEM_1head_1attention_Lap.py --device cpu --mode DP --client 3 --epochs 100 --batch_size 8 --lr 0.001 --epsilon 100 --expname Transformer_FL5_DPlap_e100 --train_data train_data_norm_log10 --test_data test_data_norm_log10



#Model Inversion Attack
python 03.attackApp_cpu.py --model model/DP_e1_modelbest.tar                       #Note: do not add "./" in front of "model/PureSGD0_modelbest.tar". Reason: 03.attackApp_cpu.py line129 model_id=model_id.split('/')[1]
python 03.attackApp_cpu_transformer.py --model model/Transformer_FL5_DPlap_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/FL40_DP_lap_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/FL20_DP_lap_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/FL10_DP_lap_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/FL5_DP_lap_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/PureSGD0_DP_gm_e100_modelbest.tar
python 03.attackApp_cpu.py --model model/FL_modelbest.tar
python 03.attackApp_cpu_transformer.py --model model/TransformerPureSGD0_modelbest.tar
python 03.attackApp_cpu_transformer.py --model model/TransformerFL_modelbest.tar
