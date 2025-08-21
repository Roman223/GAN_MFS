import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchviz import make_dot
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_float32_matmul_precision("high")

def get_ion():
    data = pd.read_csv("../data/ionosphere.data", header=None).drop(columns=[0, 1])
    data.columns = [f"feature{i}" for i in range(data.shape[1] - 1)] + ["class"]
    y = data.pop("class")
    y = y.replace({"g": 1, "b": 0}).infer_objects(copy=False)
    return data.values, y.values

def from_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

class NeuralNetwork(nn.Module):
    def __init__(self, feature_num: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_num, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x)

class SoftJointEntropy(nn.Module):
    def __init__(self, num_bins: int = 20, sigma: float = 0.001, eps: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps
        self.sigma = float(sigma)
        bin_centers = torch.linspace(0.0, 1.0, num_bins).view(num_bins, 1)
        self.register_buffer("bin_centers", bin_centers)
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
        self.register_buffer("inv_two_sigma2", torch.tensor(inv_two_sigma2))
        # Derived spacing between adjacent uniform bin centers on [0,1]
        self.delta = 1.0 / (num_bins - 1) if num_bins > 1 else None
        # Adaptive warning based on neighbor-overlap ratio κ = exp(-Δ^2/(2σ^2))
        # κ≈0 → nearly one-hot (too hard); κ≈1 → nearly uniform (too soft)
        if self.delta is not None and self.sigma > 0:
            kappa = math.exp(-(self.delta ** 2) / (2.0 * self.sigma * self.sigma))
            print(kappa)
            if kappa < 1e-17:
                warnings.warn(
                    f"SoftJointEntropy: sigma={self.sigma:.4g} yields very sharp assignments (κ≈{kappa:.2e}). "
                    f"Gradients may be high-variance. Consider increasing sigma.",
                    RuntimeWarning,
                )
            elif kappa > 0.2:
                warnings.warn(
                    f"SoftJointEntropy: sigma={self.sigma:.4g} yields very soft assignments (κ≈{kappa:.2f}). "
                    f"Bins may overlap too much; consider decreasing sigma.",
                    RuntimeWarning,
                )
        self._warned_batch = False


    @staticmethod
    def torch_crosstab(row: torch.Tensor, col: torch.Tensor, normalize="all", num_rows=None, num_cols=None):
        assert row.ndim == 1 and col.ndim == 1, f"row and col must be 1D tensors. Obtained {row.shape, col.shape}"
        assert row.shape[0] == col.shape[0], "row and col must have the same length"
        assert normalize in (None, 'all', 'index', 'columns'), "Invalid normalize argument"

        if num_rows is None:
            num_rows = int(row.max().item()) + 1
        if num_cols is None:
            num_cols = int(col.max().item()) + 1

        idx = row * num_cols + col
        flat_counts = torch.bincount(idx, minlength=num_rows * num_cols)
        crosstab = flat_counts.view(num_rows, num_cols).to(torch.float32)

        if normalize == 'all':
            total = crosstab.sum()
            if total > 0:
                crosstab = crosstab / total
        elif normalize == 'index':
            row_sums = crosstab.sum(dim=1, keepdim=True)
            crosstab = torch.where(row_sums > 0, crosstab / row_sums, torch.zeros_like(crosstab))
        elif normalize == 'columns':
            col_sums = crosstab.sum(dim=0, keepdim=True)
            crosstab = torch.where(col_sums > 0, crosstab / col_sums, torch.zeros_like(crosstab))

        return crosstab

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N,1) targets: (N,1) in {0,1}
        probs = torch.sigmoid(logits)  # (N,1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

        dist2 = (probs - self.bin_centers.T).pow(2)  # (N,B)
        weight_logits = -dist2 * self.inv_two_sigma2  # (N,B)
        weights = F.softmax(weight_logits, dim=1)  # (N,B)

        # Warn once if batch size is likely too small for stable JE estimates
        if not self._warned_batch:
            batch_size = logits.shape[0]
            if batch_size < 10 * self.num_bins:
                warnings.warn(
                    f"SoftJointEntropy: batch_size={batch_size} is small for num_bins={self.num_bins}. "
                    f"Estimator may be high-variance; aim for ≥ ~10 samples per bin.",
                    RuntimeWarning,
                )
            self._warned_batch = True

        # Aggregate counts per bin and class to estimate joint p(bin, class)
        t = targets.float()  # (N,1)
        class1 = weights.T @ t  # (B,1)
        class0 = weights.T @ (1.0 - t)  # (B,1)

        alpha = 1e-3
        counts = torch.cat([class0, class1], dim=1) + alpha

        flat = counts.reshape(-1)
        prob = flat / (flat.sum() + self.eps)
        log_p = torch.log2(prob + self.eps)
        return -(prob * log_p).sum()

class LossModule(nn.Module):
    def __init__(self, num_bins=20, sigma=0.008, lambda_entropy=0.01):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.je = SoftJointEntropy(num_bins=num_bins, sigma=sigma)
        self.lambda_entropy = lambda_entropy
        self.etr_est = []
        self.etr_true = []
        # Warn about extreme lambda choices
        if not (1e-4 <= float(lambda_entropy) <= 1.0):
            warnings.warn(
                f"LossModule: lambda_entropy={lambda_entropy} is extreme. "
                f"Try values in [1e-2, 1e-1] and adjust based on gradient ratios.",
                RuntimeWarning,
            )

    def calc_joint_ent(self,
            vec_x: np.ndarray, vec_y: np.ndarray, epsilon: float = 1.0e-8
    ):
        """Compute joint entropy between `vec_x and vec_y."""
        edges = np.linspace(0.0, 1.0, self.je.num_bins + 1)

        vec_x_binned = np.digitize(vec_x.reshape(-1), edges[:-1], right=False)

        vec_y_int = vec_y.flatten().astype(int)

        joint_prob_mat = (
                pd.crosstab(vec_y_int, vec_x_binned, normalize=True).values + epsilon
        )
        # print(pd.crosstab(vec_y_int, vec_x_binned, normalize=True).values)
        joint_prob_mat = joint_prob_mat.T

        joint_ent = np.sum(
            np.multiply(joint_prob_mat, np.log2(joint_prob_mat))
        )
        return -1.0 * joint_ent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        je_val = self.je(logits, targets)
        pfmfe_val = self.calc_joint_ent(from_tensor_to_numpy(torch.sigmoid(logits)),
                                        from_tensor_to_numpy(targets))
        # pfmfe_val = 0
        bce = self.bce(logits, targets.float())
        # return self.bce(logits, targets.float()) + self.lambda_entropy * self.je(logits, targets.float())
        return bce + self.lambda_entropy * je_val, pfmfe_val, from_tensor_to_numpy(je_val)

def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> DataLoader:
    ds = TensorDataset(X, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )

def train(
    model: nn.Module,
    loss_mod: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3
):
    losses = []
    entr_true = []
    entr_est = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_mod.to(device)

    train_dl = make_dataloader(X, y, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bar = tqdm(range(1, epochs + 1), disable=False)
    for epoch in bar:
        entropy_est_per_epochs = []
        entropy_true_per_epochs = []
        model.train()
        for step, (xb, yb) in enumerate(train_dl, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            loss, entropy_true, entropy_est = loss_mod(logits, yb)
            entropy_est_per_epochs.append(entropy_est)
            entropy_true_per_epochs.append(entropy_true)

            loss.backward()
            opt.step()

            bar.set_description(f"epoch {epoch}/{epochs}, loss: {loss.item():.4f}")
        losses.append(loss.mean().item())
        entr_est.append(np.mean(entropy_est_per_epochs))
        entr_true.append(np.mean(entropy_true_per_epochs))
    if not os.path.exists("test_Loss"):
        make_dot(loss, params=dict(model.named_parameters())).render("test_Loss", format="png")
    return model, losses, entr_true, entr_est

epochs = 500
batch_size = 64

# feature_dim = 20
# N = 1000

# X, y = make_classification(N, feature_dim,
#                            n_classes=2,
#                            n_clusters_per_class=5,
#                            n_informative=13,)
X, y = get_ion()


print(X.shape, y.shape)

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

print(X_train.shape, y_train.shape)
X = torch.from_numpy(X_train).float().to(device, non_blocking=True)
y = torch.from_numpy(y_train).float().to(device, non_blocking=True).view(-1, 1)

sigma = 0.03
n_bins = 5

lambda_entropy = 0.1
model = NeuralNetwork(feature_num=X.shape[1])
loss_mod = LossModule(num_bins=n_bins, sigma=sigma, lambda_entropy=lambda_entropy)

trained_model, losses, entropy_true, entropy_est = train(
    model,
    loss_mod,
    X,
    y,
    epochs=epochs,
    batch_size=batch_size,
    lr=1e-3
)

preds_logit = trained_model(torch.from_numpy(x_test).to(device).float()).float()
preds = torch.sigmoid(preds_logit).cpu().detach().round(decimals=4).numpy()
preds_dics = np.where(preds > 0.5, 1, 0).astype(int)
report = classification_report(y_test, preds_dics, output_dict=True)

plt.plot(losses)
plt.title(f"Loss. Lambda_entropy: {lambda_entropy:.2f}, f1: {report['macro avg']['f1-score']:.3f}")
plt.show()
plt.plot(entropy_est, label="etr_est")
plt.plot(entropy_true, label="etr_true")
plt.title(f"n_bins={n_bins}, sigma={sigma}; f1: {report['macro avg']['f1-score']:.3f}")
plt.legend()
plt.grid()
plt.show()


