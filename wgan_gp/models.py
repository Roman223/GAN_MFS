import torch
import torch.nn as nn

class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        # out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)

class Generator(nn.Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        self.latent_dim = embedding_dim
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(self, data_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        seq = []
        self.data_dim = data_dim

        dim = data_dim
        for item in list(discriminator_dim):
            # seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        return self.seq(input_.view(-1, self.data_dim))