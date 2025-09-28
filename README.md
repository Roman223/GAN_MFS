# WGAN-GP with Meta-Feature Statistics (MFS)

A PyTorch implementation of Wasserstein GAN with Gradient Penalty (WGAN-GP) enhanced with meta-feature statistics preservation for high-quality synthetic tabular data generation.

## Features

- **WGAN-GP Implementation**: Stable GAN training with Wasserstein distance and gradient penalty
- **Meta-Feature Statistics (MFS) Integration**: Preserves statistical properties of the original data
- **Comprehensive Evaluation**: Multiple utility and fidelity metrics for synthetic data assessment
- **Experiment Tracking**: Built-in Aim tracking for monitoring training progress
- **Flexible Architecture**: Configurable generator and discriminator architectures
- **Multi-Dataset Support**: Tested on various tabular datasets including Abalone and California Housing

## Architecture

The project implements two main training approaches:

1. **Vanilla WGAN-GP**: Standard WGAN-GP implementation
2. **MFS-Enhanced WGAN-GP**: WGAN-GP with additional meta-feature statistics preservation loss

### Key Components

- **Generator**: Residual network-based generator with configurable dimensions
- **Discriminator**: Multi-layer discriminator with LeakyReLU activations
- **MFS Manager**: PyTorch implementation of meta-feature extraction (correlation, covariance, eigenvalues, etc.)
- **Wasserstein Distance**: Topological distance computation for MFS alignment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Roman223/GAN_MFS
cd GAN_MFS
```

2. Install dependencies:
```bash
pip install -r req.txt
```

## Usage

### Basic Training

```python
from wgan_gp.models import Generator, Discriminator
from wgan_gp.training import Trainer, TrainerModified

# Configure hyperparameters
learning_params = {
    'epochs': 1000,
    'learning_rate_D': 0.0001,
    'learning_rate_G': 0.0001,
    'batch_size': 64,
    'gp_weight': 10,
    'generator_dim': (64, 128, 64),
    'discriminator_dim': (64, 128, 64),
    'emb_dim': 32,
}

# For MFS-enhanced training
learning_params.update({
    'mfs_lambda': [0.1, 2.0],  # or single float value
    'subset_mfs': ['mean', 'var', 'eigenvalues'],
    'sample_number': 10,
    'sample_frac': 0.5,
})

# Initialize and train model
trainer = TrainerModified(...)  # or Trainer() for vanilla WGAN-GP
trainer.train(data_loader, epochs, plot_freq=100)
```

## Data Preparation

The project expects tabular data with the target variable. Example preprocessing:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('your_data.csv')
y = data.pop('target').values
X = data.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Combine features and target for training
training_data = np.hstack([X_train, y_train.reshape(-1, 1)])
```

## Meta-Feature Statistics (MFS)

The MFS component preserves important statistical properties of the original data:

### Supported Meta-Features

- **Statistical**: mean, variance, standard deviation, range, min, max, median
- **Distributional**: skewness, kurtosis, interquartile range
- **Relational**: correlation matrix, covariance matrix, eigenvalues
- **Advanced**: sparsity, mad (median absolute deviation)

### MFS Loss Function

The generator loss combines adversarial loss with MFS preservation:

```python
g_loss = d_generated.mean() + λ * wasserstein_distance(mfs_real, mfs_synthetic)
```

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

### Utility Metrics

- **R² Score**: Regression performance preservation
- **RMSE**: Root mean squared error
- **MAPE**: Mean absolute percentage error

### Fidelity Metrics

- **Correlation Matrix Distance**: Cosine distance between correlation matrices
- **Jensen-Shannon Divergence**: Marginal distribution similarity
- **Topological Distance**: Persistent homology-based comparison

## Experiment Tracking

Aim tracking is implemented and strongly advised for monitoring:

- Training losses (Generator, Discriminator, MFS)
- Gradient norms and flow visualization
- Sample quality progression
- Comprehensive metrics logging

Instead, use:
```python
import aim
tracker = aim.Run(experiment="WGAN-GP")
tracker[hparams] = learning_params
```

## Project Structure

```
wgan_gp/
    __init__.py
    main.py                 # Main training script
    models.py               # Generator and Discriminator definitions
    training.py             # Training loops and MFS integration
    pymfe_to_torch.py       # PyTorch meta-feature extraction
    utils.py                # Utility functions and metrics
    req.txt                 # Dependencies
    gitignore               # Git ignore file
    README.md               # This file
```

## Key Dependencies

- **PyTorch**: Deep learning framework
- **Aim**: Experiment tracking
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **torch-topological**: Topological data analysis
- **POT**: Optimal transport (Wasserstein distance)
- **pymfe**: Meta-feature extraction
- **xgboost**: Gradient boosting for utility metrics

## Documentation

This project includes comprehensive self-hosted documentation built with MkDocs and Material theme. The documentation provides detailed API references, usage examples, and theoretical background.

### Building and Running Documentation Locally

To build and serve the documentation locally:

1. **Install documentation dependencies** (if not already installed):
```bash
pip install mkdocs-material 'mkdocstrings[python]'
```

2. **Serve the documentation** locally:
```bash
mkdocs serve --config-file osa_mkdocs.yml
```

3. **Access the documentation** at `http://localhost:8000`

The documentation includes:
- **API Reference**: Auto-generated from docstrings using mkdocstrings
- **Module Documentation**: Detailed explanations of each component
- **Usage Examples**: Code samples and tutorials
- **Meta-Feature Statistics Table**: Complete reference of all available MFS features

### Building Static Documentation

To build static HTML files for deployment:

```bash
mkdocs build -f osa_mkdocs.yml
```

The generated documentation will be available in the `site/` directory and can be deployed to any static hosting service.

## Configuration

Key hyperparameters can be configured in `main.py`:

```python
learning_params = dict(
    epochs=1000,                    # Training epochs
    learning_rate_D=0.0001,         # Discriminator learning rate
    learning_rate_G=0.0001,         # Generator learning rate
    batch_size=64,                  # Batch size
    gp_weight=10,                   # Gradient penalty weight
    generator_dim=(64,128,64),      # Generator architecture
    discriminator_dim=(64,128,64),  # Discriminator architecture
    mfs_lambda=0.1,           # MFS loss weights
    subset_mfs=['mean','var'],      # Selected meta-features
    sample_number=10,               # Number of variates for MFS
    sample_frac=0.5,                # Fraction for variate sampling
)
```

## Results

The model generates high-quality synthetic tabular data that preserves:

- Statistical distributions of original features
- Correlational structure between variables
- Utility for downstream machine learning tasks
- Topological properties of the data manifold

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wgan_gp_mfs,
    title={WGAN-GP with Meta-Feature Statistics for Synthetic Tabular Data Generation},
    author={[Your Name]},
    year={2024},
    url={https://github.com/your-username/your-repo}
}
```

## Acknowledgments

- Original WGAN-GP paper: Improved Training of Wasserstein GANs  
- PyMFE library for meta-feature extraction inspiration  
- Aim team for excellent experiment tracking tools
