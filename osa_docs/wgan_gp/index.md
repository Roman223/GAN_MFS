# WGAN-GP Module

## Overview

This module implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) tailored for synthetic data generation. It includes components for defining datasets, generator and discriminator models, meta-feature extraction using PyTorch, and custom training procedures. The module provides tools for incorporating Marginal Feature Significance (MFS) into the training loop, enabling targeted feature selection and manipulation to improve the utility of the generated data.

## Purpose

The primary purpose of this module is to generate synthetic datasets that closely mimic real-world data, with a specific focus on preserving statistical properties relevant to downstream tasks. It provides a framework for training GANs with gradient penalty to stabilize training and incorporates MFS to allow users to guide the GAN towards generating data that preserves the significance of specific features. This is particularly useful for data augmentation, privacy preservation, and creating representative datasets where real data is limited or sensitive, and where maintaining the utility of specific features is crucial.
