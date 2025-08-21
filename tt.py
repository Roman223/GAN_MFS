import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sample_from_Xy_tensors(X:torch.Tensor, n_samples):
    indices = torch.randperm(X.size(0))
    indices_trunc = indices[:n_samples]
    X_sampled_tensor = X[indices_trunc[:n_samples]]
    # y_sampled_tensor = y[indices_trunc[:n_samples]]
    # return X_sampled_tensor, y_sampled_tensor
    return X_sampled_tensor

def create_variates(X: torch.Tensor, y: torch.Tensor=None, sample_frac=0.3, sample_number=100):
    assert sample_frac < 1
    total_size = X.size(0)
    n_samples = int(total_size * sample_frac)
    return [sample_from_Xy_tensors(X, n_samples) for _ in range(sample_number)]

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def ft_eigenvals(x: torch.Tensor) -> torch.Tensor:
    # taking real part of first two eigenvals
    centered = x - x.mean(dim=0, keepdim=True)
    covs = cov(centered, rowvar=False)
    return torch.linalg.eigvalsh(covs)

def calculate_mfs_torch(X: torch.Tensor
                        , y: torch.Tensor = None) -> torch.Tensor:
    return get_mfs(X, y)

def reshape_mfs_from_variates(mfs_from_variates: list):
    stacked = torch.stack(mfs_from_variates)
    reshaped = stacked.transpose(0, 1)
    return reshaped

def get_mfs(X, y):
    mfs = [
            ft_eigenvals(X),
        # ... there are other mfs (truncated)
        ]
    # shapes = [i.shape.numel() for i in mfs]
    # mfs = [self.pad_only(mf, max(shapes)) for mf in mfs]
    return torch.stack(mfs)

def create_joint(plot=False, sample_size=1000,
                 arc_size=1000):
    # Generate two Gaussian blobs
    mean1 = [3, 3]
    mean2 = [-5, -5]
    cov = [[0.2, 0], [0, 0.2]]
    samples1 = np.random.multivariate_normal(mean1, cov, size=sample_size)
    samples2 = np.random.multivariate_normal(mean2, cov, size=sample_size)

    # Create a perturbed arc (half-circle)
    theta = np.linspace(0, 2 * np.pi, arc_size)
    arc_x = np.cos(theta)
    arc_y = np.sin(theta)
    arc = np.stack([arc_x, arc_y], axis=1)

    # Translate and add noise to arc
    arc = 5.5 * arc  # scale to match distance
    arc[:, 0] -= 1 # center between the two blobs
    arc[:, 1] -= 1  # center between the two blobs
    arc += np.random.normal(scale=0.2, size=arc.shape)

    # Combine all points
    all_samples = np.vstack([samples1, samples2, arc])

    return torch.tensor(all_samples, dtype=torch.float)

X = create_joint(sample_size=1500, arc_size=2000, plot=False)
variates = create_variates(X,
                           sample_number=100,
                           sample_frac=0.5, )

mfs_distr = [calculate_mfs_torch(X_sample) for X_sample in variates]  # list of Tensors
mfs_distr = reshape_mfs_from_variates(mfs_distr)

mfs_distr_real = calculate_mfs_torch(X)
real_eigvals = mfs_distr_real[0].cpu().detach().numpy()

eigvals = mfs_distr[0].cpu().numpy()
eigvals = pd.DataFrame(eigvals[:, :2], columns=["dim1", "dim2"])

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
p = sns.jointplot(data=eigvals, x="dim1", y="dim2", kind="kde",
                  alpha=0.6)
p.fig.suptitle("eigvals")
plt.scatter(*real_eigvals, marker="*", color="red", s=100, label="real eigvals")
plt.tight_layout()

plt.show()