# PP-GenomeDL
Privacy-Preserving Genome Deep Learning
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
