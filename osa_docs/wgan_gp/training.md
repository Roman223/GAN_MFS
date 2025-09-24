# Training Module

This module contains the core training classes for WGAN-GP with Meta-Feature Statistics (MFS) preservation.

## Classes

### Trainer
Base trainer class for vanilla WGAN-GP implementation with gradient penalty.

### TrainerModified  
Enhanced trainer class that incorporates Meta-Feature Statistics preservation during training.

## Key Features

- **Wasserstein Distance with Gradient Penalty**: Stable GAN training using WGAN-GP formulation
- **Meta-Feature Statistics Preservation**: Maintains statistical properties of original data
- **Flexible Loss Weighting**: Configurable balance between adversarial and MFS losses
- **Comprehensive Monitoring**: Gradient flow visualization and training metrics tracking
- **Experiment Tracking**: Built-in Aim integration for training progress monitoring

::: wgan_gp.training