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

## Repository Structure

```
PP-GenomeDL/
├── data/                           # Data directory
│   ├── raw/                        # Raw TCGA data files (.npy)
│   └── processed/                  # Preprocessed and normalized data
├── models/                         # Deep learning model architectures
│   ├── naive_model.py              # Fully connected neural network (4-layer)
│   ├── transformer_model.py        # Transformer model based on T-GEM
│   ├── cnn_qat_model.py            # QAT CNN for FHE compatibility
│   └── DL_modules/                 # User-defined model directory (for platform)
├── privacy/                        # Privacy-preserving implementations
│   ├── differential_privacy/
│   │   ├── gaussian.py             # Gaussian mechanism
│   │   ├── analytic_gaussian.py    # Analytic Gaussian mechanism
│   │   ├── correlated_gaussian.py  # Correlated Gaussian mechanism
│   │   └── laplace.py              # Laplace mechanism
│   ├── federated_learning/
│   │   ├── fl_server.py            # FL aggregation server
│   │   ├── fl_client.py            # FL client with local training
│   │   └── fl_coordinator.py       # FL orchestration and communication
│   ├── fhe/
│   │   ├── fhe_compile.py          # Concrete-ML FHE compilation
│   │   ├── fhe_inference.py        # Encrypted inference pipeline
│   │   └── fhe_dp_integration.py   # FHE + DP combined model
│   └── fl_mp_ckks_fhe/
│       ├── framework.py            # Core FL-MP-CKKS-FHE framework
│       ├── mp_ckks.py              # Multi-party CKKS encryption (OpenFHE)
│       ├── adaptive_scaling.py     # Adaptive per-chunk scaling strategy
│       ├── simd_optimization.py    # SIMD ciphertext packing
│       ├── gradient_compression.py # Top-K gradient sparsification
│       └── utils.py                # Gradient clipping, normalization utilities
├── attacks/                        # Privacy attack testing
│   ├── model_inversion_attack.py   # MIA implementation
│   ├── mia_visualization.py        # Z-score distribution visualization
│   └── mia_metrics.py              # MSE, NRMSE, PSNR, SSIM evaluation
├── platform/                       # Privacy-preserving deep learning platform
│   ├── run_platform.py             # Main entry point
│   ├── config.ini                  # Configuration file template
│   └── README.md                   # Platform-specific documentation
├── experiments/                    # Experiment scripts
│   ├── exp1_dp_fl_naive.py         # DP + FL on naïve model
│   ├── exp2_dp_fl_transformer.py   # DP + FL on transformer model
│   ├── exp3_fhe_cnn.py             # FHE-compatible CNN
│   ├── exp4_fhe_dp_cnn.py          # FHE + DP CNN with MIA testing
│   ├── exp5_fl_mp_ckks_fhe.py      # FL-MP-CKKS-FHE framework
│   ├── exp6_adaptive_scaling.py    # Adaptive scaling evaluation
│   ├── exp7_simd.py                # SIMD optimization evaluation
│   └── exp8_gradient_compression.py# Top-K compression evaluation
├── results/                        # Output directory for results and logs
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

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

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

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

The dataset used in this study consists of RNA-seq gene expression profiles from The Cancer Genome Atlas (TCGA), downloaded via the GDC Data Portal using TCGA-assembler2.

```bash
# Download and preprocess TCGA data
python data/download_tcga.py

# After preprocessing:
# - 13,057 samples, 20,531 gene expression features
# - 37 cancer types
# - Log10-scaled and normalized
# - 80/20 train/test split (default)
```

The processed data should be stored in NumPy format (`.npy`) with the following structure:

```python
{
    'sample1_name': {
        'type': 'BRCA',
        'features': array([0., 0.464, 0.498, ..., 0.742, 0.631, 0.381])
    },
    'sample2_name': {
        'type': 'STES',
        'features': array([0.424, 0.519, 0.522, ..., 0.688, 0.570, 0.])
    },
    ...
}
```

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

## Key Results

| Model / Framework | Accuracy | Privacy Features | Notable Metric |
|-------------------|----------|------------------|----------------|
| Naïve DL (baseline) | 75% | None | — |
| Naïve DL + DP + FL | 72–75% | DP (ε=100) + FL (5 clients) | Resists MIA |
| Transformer + DP + FL | 72% | DP (ε=100) + FL (5 clients) | Resists MIA |
| FHE-CNN (29 cancers) | 96.14% | FHE encrypted inference | ~1,456 s/sample |
| FHE-CNN + DP | >90% | FHE + DP (ε=10–800) | Enhanced MIA resistance |
| FL-MP-CKKS-FHE | 70% (test) | MP-FHE + FL (3 clients) | First of its kind |
| + Adaptive Scaling | 72.66% | MP-FHE + FL | 42.2% faster |
| + SIMD Optimization | 74.89% | MP-FHE + FL | 72.2% faster (3.6× speedup) |
| + Top-K 10% Compression | 69.53% | MP-FHE + FL | 84.8% faster |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{dong2026ppgenomedl,
    title     = {Toward Secure Genomic Intelligence: Privacy-Preserving Deep Learning
                 Frameworks for Cancer Classification and Collaborative Analysis},
    author    = {Dong, Yucheng},
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
