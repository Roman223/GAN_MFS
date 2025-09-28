# PyMFE to Torch Module

This module provides a PyTorch implementation of Meta-Feature Extraction (MFE) for statistical analysis of tabular data.

## Overview

The PyMFE to Torch module converts meta-feature extraction operations into PyTorch tensors, enabling differentiable computation of statistical properties during GAN training. This is essential for the Meta-Feature Statistics (MFS) preservation component of the WGAN-GP implementation.

## Key Features

### Statistical Meta-Features
- **Correlation**: Pearson correlation coefficients between features
- **Covariance**: Covariance matrix computation
- **Eigenvalues**: Principal component eigenvalues for dimensionality analysis
- **Distributional Statistics**: Mean, variance, standard deviation, range, min, max
- **Advanced Statistics**: Skewness, kurtosis, interquartile range, sparsity

### PyTorch Integration
- **Differentiable Operations**: All computations maintain gradient flow
- **GPU Acceleration**: CUDA-compatible tensor operations
- **Batch Processing**: Efficient computation over data batches
- **Device Management**: Automatic device placement for tensors

### MFEToTorch Class
The main class that provides:
- Feature method mapping for easy access to statistical functions
- Torch-native implementations of traditional meta-feature extraction
- Integration with the training loop for real-time MFS computation
- Support for subset feature selection for targeted preservation

## Usage in Training

This module is crucial for the MFS-enhanced WGAN-GP training, where it:
1. Computes meta-features for real data variates
2. Calculates corresponding features for generated synthetic data
3. Enables Wasserstein distance computation between feature distributions
4. Provides gradients for generator optimization

::: wgan_gp.pymfe_to_torch