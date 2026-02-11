# PP-GenomeDL

**Toward Secure Genomic Intelligence: Privacy-Preserving Deep Learning Frameworks for Cancer Classification and Collaborative Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

PP-GenomeDL is the official code repository for the dissertation:

> **Toward Secure Genomic Intelligence: Privacy-Preserving Deep Learning Frameworks for Cancer Classification and Collaborative Analysis**

This repository provides implementations of privacy-preserving deep learning models and frameworks for cancer genomic data analysis. It includes deep learning models integrated with differential privacy (DP), federated learning (FL), fully homomorphic encryption (FHE), and the novel FL-MP-CKKS-FHE framework — the first integration of multi-party fully homomorphic encryption with federated learning for deep learning model training on genomic data.

### Key Features

- **Differential Privacy (DP):** Four noise mechanisms (Gaussian, Analytic Gaussian, Correlated Gaussian, Laplace) with configurable privacy budgets
- **Federated Learning (FL):** Custom client-server architecture with configurable distributed clients (5, 10, 20, 40 agents)
- **FHE-Compatible CNN:** Quantization-aware training (QAT) and FHE compilation for encrypted inference on genomic data
- **FHE + DP Integration:** Dual-protection deep learning model combining FHE with differential privacy
- **FL-MP-CKKS-FHE Framework:** Multi-party CKKS homomorphic encryption with federated learning for secure collaborative model training
- **Algorithm Optimizations:** Adaptive per-chunk scaling, SIMD packing, and Top-K gradient compression for CKKS-based FL
- **Model Inversion Attack (MIA) Testing:** Comprehensive attack evaluation with quantitative metrics (MSE, NRMSE, PSNR, SSIM)
- **Privacy-Preserving Platform:** A configurable platform for applying privacy-preserving features to user-defined deep learning models

---


## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenFHE library (for FL-MP-CKKS-FHE framework)

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/PP-GenomeDL.git
cd PP-GenomeDL


### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning framework |
| Brevitas | ≥ 0.12.1 | Quantization-aware training |
| Concrete-ML | latest | FHE compilation |
| OpenFHE (Python wrapper) | latest | Multi-party CKKS-FHE |
| scikit-learn | ≥ 1.0 | Grid search, metrics |
| skorch | ≥ 0.12 | PyTorch-sklearn wrapper |
| NumPy | ≥ 1.21 | Numerical computation |
| Matplotlib | ≥ 3.5 | Visualization |
| scikit-image | ≥ 0.19 | SSIM metric computation |

---

## Data Preparation

### TCGA Pan-Cancer Data

The dataset used in this study consists of RNA-seq gene expression profiles from The Cancer Genome Atlas (TCGA).             
The data can be downloaded from Zenodo (https://doi.org/10.5281/zenodo.18603490)                
These data include:        
# - 13,057 samples, 20,531 gene expression features               
# - 37 cancer types                   
# - Log10-scaled and normalized                   
# - 80/20 train/test split (default)                       

---

## Usage

### 1. Cancer Type Classification with DP and FL

```python
# Naïve deep learning model with differential privacy
python experiments/exp1_dp_fl_naive.py \
    --noise_algorithm laplace \
    --epsilon 100 \
    --fl_clients 5 \
    --epochs 200

# Transformer model with DP and FL
python experiments/exp2_dp_fl_transformer.py \
    --epsilon 100 \
    --fl_clients 5
```

**Configurable parameters:**
- `--noise_algorithm`: `gaussian`, `analytic_gaussian`, `correlated_gaussian`, `laplace`
- `--epsilon`: Privacy budget (1, 10, 50, 100, 150, 200, 300, 400)
- `--fl_clients`: Number of federated learning clients (5, 10, 20, 40)
- `--epochs`: Training epochs (200 or 400 recommended)

### 2. FHE-Compatible CNN for Encrypted Inference

```python
# Train QAT CNN and compile to FHE
python experiments/exp3_fhe_cnn.py \
    --cancer_types 29 \
    --batch_size 80 \
    --dropout 0.4 \
    --kernel_size 50

# FHE-CNN with differential privacy
python experiments/exp4_fhe_dp_cnn.py \
    --epsilon 10 \
    --epochs 40
```

### 3. FL-MP-CKKS-FHE Framework

```python
# Run the multi-party FHE federated learning framework
python experiments/exp5_fl_mp_ckks_fhe.py \
    --clients 3 \
    --epochs 10 \
    --ring_dimension 65536 \
    --scale_factor 10 \
    --clip_value 10 \
    --learning_rate 0.0001
```

### 4. Algorithm Optimizations

```python
# Adaptive per-chunk scaling
python experiments/exp6_adaptive_scaling.py \
    --scale_range 50 200

# SIMD optimization
python experiments/exp7_simd.py

# Top-K gradient compression
python experiments/exp8_gradient_compression.py \
    --topk_ratio 0.10  # Options: 0.50, 0.10, 0.05, 0.01
```

### 5. Model Inversion Attack Testing

```python
# Run MIA on a trained model
python attacks/model_inversion_attack.py \
    --model_path results/model.pth \
    --cancer_types COADREAD KICH UVM \
    --metrics mse nrmse psnr ssim
```

### 6. Privacy-Preserving Platform

```bash
# 1. Place your PyTorch model in platform/DL_modules/ (class name must be "Model")
# 2. Store training/test data in platform/data/ (.npy format)
# 3. Configure platform/config.ini
# 4. Run:
python platform/run_platform.py --config platform/config.ini
```

---


## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{dong2026ppgenomedl,
    title     = {Toward Secure Genomic Intelligence: Privacy-Preserving Deep Learning
                 Frameworks for Cancer Classification and Collaborative Analysis},
    author    = {Dong, Yibo},
    year      = {2026},
    school    = {Mississippi State University},
    department = {Department of Computer Science}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- TCGA and GDC for the pan-cancer genomic dataset
- OpenFHE team for the multi-party CKKS-FHE library
- Zama for the Concrete-ML and Brevitas frameworks
- PPML-Omics for the model inversion attack testing baseline

---

## Contact

Yibo Dong — Mississippi State University, Department of Computer Science

For questions about the code or the dissertation, please open an issue or contact via email.
