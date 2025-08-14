import os.path

from sklearn.model_selection import train_test_split
from torchviz import make_dot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")

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
        bin_centers = torch.linspace(0.0, 1.0, num_bins).view(1, num_bins)
        self.register_buffer("bin_centers", bin_centers)
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
        self.register_buffer("inv_two_sigma2", torch.tensor(inv_two_sigma2))


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
        probs = torch.sigmoid(logits)
        dist2 = (probs - self.bin_centers) ** 2
        weight_logits = -dist2 * self.inv_two_sigma2
        weights = F.softmax(weight_logits, dim=-1)

        t = targets.float()
        WT = weights.transpose(0, 1)
        class1 = WT @ t
        class0 = WT @ (1.0 - t)
        counts = torch.cat([class0, class1], dim=1)
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
        self.etrr = []

    def calc_joint_ent(self,
            vec_x: np.ndarray, vec_y: np.ndarray, epsilon: float = 1.0e-8
    ):
        # print("enter pymfe")
        """Compute joint entropy between `vec_x and vec_y."""
        discretizer = KBinsDiscretizer(strategy='uniform', encode='ordinal', n_bins=self.je.num_bins)

        vec_x_binned = discretizer.fit_transform(vec_x.reshape(-1,1)).astype(int).flatten()
        vec_y_int = vec_y.flatten().astype(int)

        joint_prob_mat = (
                pd.crosstab(vec_y_int, vec_x_binned, normalize=True).values + epsilon
        )
        # print(pd.crosstab(vec_y_int, vec_x_binned, normalize=True).values + epsilon)
        joint_prob_mat = joint_prob_mat.T

        joint_ent = np.sum(
            np.multiply(joint_prob_mat, np.log2(joint_prob_mat))
        )
        # print("quit pymfe")
        return -1.0 * joint_ent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # je_val = self.je(logits, targets)
        # pfmfe_val = self.calc_joint_ent(logits.cpu().detach().numpy(), targets.cpu().detach().numpy())
        # print(f"\nhard: {pfmfe_val} soft: {je_val} delta: {pfmfe_val - je_val}")
        # print(f"Expected: {logits.shape} {targets.shape}")
        # entropy_uncert = self.je(logits, targets) - \
        #         self.calc_joint_ent(logits.cpu().detach().numpy(), targets.cpu().detach().numpy()).item()
        # self.etrr.append(entropy_uncert.item())
        return self.bce(logits, targets.float()) + self.lambda_entropy * self.je(logits, targets.float())

def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 256, num_workers: int | None = None) -> DataLoader:
    ds = TensorDataset(X, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
    )

@torch.no_grad()
def evaluate(model: nn.Module, loss_mod: nn.Module, dl: DataLoader) -> float:
    model.eval()
    total, count = 0.0, 0
    for xb, yb in dl:
        xb = xb.to(next(model.parameters()).device, non_blocking=True)
        yb = yb.to(next(model.parameters()).device, non_blocking=True)
        loss = loss_mod(model(xb), yb)
        total += loss.item()
        count += 1
    return total / max(1, count)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_mod.to(device)

    train_dl = make_dataloader(X, y, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bar = tqdm(range(1, epochs + 1))
    for epoch in bar:
        model.train()
        for step, (xb, yb) in enumerate(train_dl, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_mod(logits, yb)

            loss.backward()
            opt.step()

            bar.set_description(f"epoch {epoch}/{epochs}, loss: {loss.item():.4f}")
        losses.append(loss.mean().item())
    if not os.path.exists("test_Loss"):
        make_dot(loss, params=dict(model.named_parameters())).render("test_Loss", format="png")
    return model, losses

# X: (N, feature_dim), y: (N, 1) in {0,1}
feature_dim = 5
epochs = 500
batch_size = 200

N = 1000
# X = (torch.randn(N, feature_dim, device=device) + 5.0)
# y = torch.randint(0, 2, (N, 1), dtype=torch.long, device=device)

X, y = make_classification(N, feature_dim, n_classes=2, )

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.shape, y.shape)
X = torch.from_numpy(X_train).float().to(device, non_blocking=True)
y = torch.from_numpy(y_train).float().to(device, non_blocking=True).view(-1, 1)

model = NeuralNetwork(feature_num=X.shape[1])
loss_mod = LossModule(num_bins=3, sigma=0.0002, lambda_entropy=.1)

trained_model, losses = train(
    model,
    loss_mod,
    X,
    y,
    epochs=epochs,
    batch_size=batch_size,
    lr=1e-2
)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("LOss")
plt.show()
from sklearn.metrics import classification_report

preds_logit = trained_model(torch.from_numpy(x_test).to(device).float()).float()
preds = F.sigmoid(preds_logit).cpu().detach().round(decimals=4).numpy()
preds_dics = np.where(preds > 0.5, 1, 0).astype(int)

print(classification_report(y_test, preds_dics))