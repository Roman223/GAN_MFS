# WGAN-GP Module

## Overview

This module implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) tailored for synthetic tabular data generation. It includes components for defining datasets, generator and discriminator models, meta-feature extraction using PyTorch, and custom training procedures. The module provides tools for incorporating Meta-Feature Statistics (MFS) into the training loop, enabling the preservation of statistical properties such as correlation, covariance, eigenvalues, mean, and variance to improve the fidelity and utility of the generated synthetic data.

## Purpose

The primary purpose of this module is to generate high-quality synthetic tabular datasets that closely mimic real-world data distributions while preserving essential statistical characteristics. It provides a framework for training GANs with gradient penalty to stabilize training and incorporates MFS loss components to ensure the synthetic data maintains the same meta-feature distributions as the original data. This approach is particularly valuable for:

- **Data Augmentation**: Creating additional training samples while preserving data characteristics
- **Privacy Preservation**: Generating synthetic datasets that maintain utility without exposing sensitive information  
- **Research and Development**: Providing representative datasets when real data access is limited or restricted
- **Model Testing**: Creating controlled datasets with known statistical properties for algorithm validation

## Key Components

The WGAN-GP module consists of several interconnected components:

### Core Architecture
- **Generator**: Residual network-based architecture that transforms random noise into synthetic data samples
- **Discriminator**: Multi-layer network that distinguishes between real and synthetic data using Wasserstein distance
- **Gradient Penalty**: Regularization technique that enforces Lipschitz constraint for stable training

### Meta-Feature Statistics Integration
- **MFE PyTorch Implementation**: Custom PyTorch implementation of meta-feature extraction
- **Statistical Preservation**: Maintains correlation matrices, covariance structures, eigenvalues, and distributional properties
- **Wasserstein Distance Matching**: Uses optimal transport theory to align meta-feature distributions between real and synthetic data

### Training Infrastructure
- **Dual Training Modes**: Support for both vanilla WGAN-GP and MFS-enhanced training
- **Flexible Loss Functions**: Configurable weighting between adversarial loss and meta-feature preservation
- **Experiment Tracking**: Built-in Aim integration for comprehensive training monitoring

This module enables researchers and practitioners to generate synthetic data that not only looks realistic but also preserves the underlying statistical structure necessary for downstream machine learning tasks.
