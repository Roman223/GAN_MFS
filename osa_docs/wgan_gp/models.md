# Models Module

This module defines the neural network architectures for the WGAN-GP implementation.

## Architecture Overview

The models are designed for tabular data generation with residual connections to improve training stability and gradient flow.

## Classes

### Residual
Building block layer that applies linear transformation, batch normalization, and ReLU activation with residual connections.

### Generator
Residual network-based generator that transforms random noise into synthetic tabular data samples.

### Discriminator
Multi-layer discriminator network that evaluates the quality of generated samples using Wasserstein distance.

## Key Features

- **Residual Connections**: Improved gradient flow and training stability
- **Configurable Architecture**: Flexible layer dimensions for different dataset sizes
- **Batch Normalization**: Stabilized training dynamics
- **Leaky ReLU Activations**: Better gradient propagation in discriminator

::: wgan_gp.models