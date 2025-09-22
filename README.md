# GAN_MFS

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

GAN_MFS introduces a novel approach to synthetic data generation, focusing on creating datasets that closely mirror real-world data while preserving the utility of specific features. It addresses the challenge of generating high-quality synthetic data, particularly when real data is limited or sensitive. The core innovation lies in incorporating Marginal Feature Significance (MFS) into the training of Wasserstein GANs with Gradient Penalty (WGAN-GP), guiding the generator to maintain the significance of targeted features. This methodology, supported by aim tracking for monitoring training progress, enhances the fidelity of synthetic data by aligning meta-feature distributions between real and synthetic datasets. The project contributes to research on improving synthetic data generation techniques, demonstrating enhanced downstream utility and correlation alignment compared to existing methods, as highlighted in the associated research paper.

---

## Table of Contents

- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Citation](#citation)

---
## Content

The GAN_MFS project centers on generating synthetic tabular data using a Wasserstein GAN with Gradient Penalty (WGAN-GP), enhanced by Marginal Feature Significance (MFS). The core objective is to create synthetic datasets that closely mirror real-world data, particularly in preserving the utility of specific features relevant to downstream tasks. The architecture includes generator and discriminator models, a training loop with gradient penalty for stable training, and MFS-based feature selection. Meta-feature extraction is used to compute statistical descriptors, guiding the generator to align distributions between real and synthetic data. Aim tracking is implemented to monitor training progress. This approach supports data augmentation and privacy preservation, creating high-quality synthetic data for various applications.

---

## Algorithms

The project implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to generate synthetic tabular data. The WGAN-GP framework stabilizes GAN training using a gradient penalty, ensuring the discriminator adheres to the Lipschitz constraint, which is crucial for reliable synthetic data generation. To enhance the utility of the generated data, Marginal Feature Significance (MFS) is incorporated. MFS guides the generator to preserve the significance of specific features by aligning meta-feature distributions between real and synthetic data using the Wasserstein distance. This targeted feature selection improves the fidelity and relevance of the synthetic data for downstream tasks, such as data augmentation and privacy preservation.

---

## Installation

Install GAN_MFS using one of the following methods:

**Build from source:**

1. Clone the GAN_MFS repository:
```sh
git clone https://github.com/Roman223/GAN_MFS
```

2. Navigate to the project directory:
```sh
cd GAN_MFS
```

---

## Getting Started

Since [Aim tracking](https://aimstack.readthedocs.io/en/latest/) is implemented, it is strongly advised to utilize it for tracking and visualizing the training process. Please refer to the Aim documentation for detailed instructions on how to integrate and use Aim within your project.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    Roman223 (2025). GAN_MFS repository [Computer software]. https://github.com/Roman223/GAN_MFS

### BibTeX format:

    @misc{GAN_MFS,

        author = {Roman223},

        title = {GAN_MFS repository},

        year = {2025},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/Roman223/GAN_MFS.git}},

        url = {https://github.com/Roman223/GAN_MFS.git}

    }

---
