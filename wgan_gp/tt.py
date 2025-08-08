import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from torchviz import make_dot
from sklearn.preprocessing import KBinsDiscretizer
torch.manual_seed(5)

# Toy dataset (assumed simple tensor-based dataset)
class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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

def calc_joint_entropy(crosstab, epsilon: float = 1e-8) -> torch.Tensor:
    joint_prob = crosstab + epsilon
    joint_entropy = -torch.sum(joint_prob * torch.log2(joint_prob))
    return joint_entropy

class NeuralNetwork(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_num, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # Predicting 1 value (regression)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def custom_loss_fn(preds, targets, lambda_entropy=1.0):
    preds_numpy = preds.clone().numpy()

    discretizer = KBinsDiscretizer(strategy='uniform', encode='ordinal')
    preds_disc = torch.from_numpy(discretizer.fit_transform(preds_numpy)).int()
    preds_disc.grad = preds.grad

    joint_entropy = []
    for i in range(preds_disc.shape[1]):
        crosstab = torch_crosstab(preds_disc[:, i], targets.flatten(), normalize='all')
        crosstab.requires_grad_(True)
        joint_entropy.append(calc_joint_entropy(crosstab))

    joint_entropy = torch.stack(joint_entropy)
    mse_loss = nn.functional.mse_loss(preds, targets)
    mse_loss.requires_grad_(True)
    return mse_loss + lambda_entropy * joint_entropy

# Create synthetic data
num_rows = 1000
batch_size = 3

# data_disc = torch.randint(0, 100, size=(num_rows, 2), dtype=torch.int64)
data_cont = torch.randn(size=(num_rows, 5)) + 5
# target = torch.randn(size=(num_rows, 1)) + 5
target = torch.randint(0, 2, size=(num_rows, 1), dtype=torch.int64)
print(data_cont.shape, target.shape)
data = torch.concat([data_cont, target], dim=1)

print(data)
custom_loss = custom_loss_fn(data_cont, target)
print(custom_loss)
make_dot(custom_loss, show_attrs=True).render("test_Loss", format="png")
raise Exception
def calc_joint_ent(
    vec_x: np.ndarray, vec_y: np.ndarray, epsilon: float = 1.0e-8
):
    """Compute joint entropy between `vec_x and vec_y."""
    joint_prob_mat = (
        pd.crosstab(vec_y, vec_x, normalize=True).values + epsilon
    )

    joint_ent = np.sum(
        np.multiply(joint_prob_mat, np.log2(joint_prob_mat))
    )

    return -1.0 * joint_ent

dataset = ToyDataset(data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

feature_num = data.shape[1] - 1  # Last column is target
model = NeuralNetwork(feature_num=feature_num)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()
losses = []

bar = tqdm(range(100))
# Training loop
for i in bar:
    for batch in data_loader:
        optimizer.zero_grad()
        x_batch = batch[:, :-1]
        y_batch = batch[:, -1].unsqueeze(1)

        # Split discrete from continuous (first 2 columns are discrete)
        x_discrete = x_batch[:, :2].int()

        # Forward pass
        preds = model(x_batch)

        # Custom loss
        loss = custom_loss_fn(preds, y_batch, x_discrete)
        make_dot(loss, show_attrs=True).render("test_rest", format="png")
        bar.desc = f"loss = {round(loss.item(), 5)}"
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

import matplotlib.pyplot as plt
plt.plot(losses)
plt.grid()
plt.show()