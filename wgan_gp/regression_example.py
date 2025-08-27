import os.path
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from torchviz import make_dot
import math
import torch.nn.functional as F
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_float32_matmul_precision("high")


def from_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class NeuralNetwork(nn.Module):
    def __init__(self, feature_num: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_num, 16),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class EntropyPrecomputer:
    def __init__(self, bins,  sigma: float = 0.01):
        self.bins = bins
        self.sigma = float(sigma)
        self.columns = []
        self.discrete_columns = []

        self.feature_mins = None
        self.feature_maxs = None
        self.y_min = None
        self.y_max = None

    @property
    def feature_ranges(self):
        return f"""
        feature_mins: {self.feature_mins}
        feature_maxs:{self.feature_maxs}
        y_min: {self.y_min}
        y_max: {self.y_max}
        ________"""

    def numpy_marginal_entropy(self, x):
        rng_x = self.y_max - self.y_min
        preds01 = np.clip((x.ravel() - self.y_min) / rng_x, 0.0, 1.0)
        H, _ = np.histogram(preds01, bins=self.num_bins, range=[0, 1])
        H = H.astype(np.float64) + 1e-3
        P = H / (H.sum() + 1e-12)
        logP = np.log2(P + 1e-12)
        return float(-(P * logP).sum())

    def numpy_joint_entropy(
            self,
            preds: np.ndarray,
            targets: np.ndarray,
            # y_range: list[float],
            # x_range: list[float],
            alpha: float = 1e-3,
    ) -> float:
        if isinstance(self.feature_mins, torch.Tensor):
            feature_mins = from_tensor_to_numpy(self.feature_mins)
        else:
            feature_mins = self.feature_mins
            
        if isinstance(self.feature_maxs, torch.Tensor):
            feature_maxs = from_tensor_to_numpy(self.feature_maxs)
        else:
            feature_maxs = self.feature_maxs

        if isinstance(preds, torch.Tensor):
            preds = from_tensor_to_numpy(preds)
        if isinstance(targets, torch.Tensor):
            targets = from_tensor_to_numpy(targets)

        # x_min, x_max = x_range
        # y_min, y_max = y_range

        # Map preds to [0,1] via sigmoid and normalize targets to [0,1]
        preds01 = 1.0 / (1.0 + np.exp(-preds.reshape(-1)))
        # print(self.feature_maxs - self.feature_mins)
        # rng_x = max(self.feature_maxs - self.feature_mins, 1e-8)
        # rng_x = feature_maxs - feature_mins

        # preds01 = np.clip((preds.ravel() - feature_mins) / rng_x, 0.0, 1.0)

        rng_t = max(self.y_max - self.y_min, 1e-8)
        targets01 = np.clip((targets.reshape(-1) - self.y_min) / rng_t, 0.0, 1.0)

        H, xedges, yedges = np.histogram2d(preds01, targets01, bins=self.bins, range=[[0, 1], [0, 1]])
        H = H.astype(np.float64) + alpha
        P = H / (H.sum() + 1e-12)
        logP = np.log2(P + 1e-12)
        return float(-(P * logP).sum())

    def set_max_min(self, X_train, y_train):
        # print(type(X_train), type(y_train))
        if isinstance(X_train, (np.ndarray, pd.DataFrame)):
            self.feature_mins = np.min(X_train, axis=0).astype(np.float32)
            self.feature_maxs = np.max(X_train, axis=0).astype(np.float32)
        if isinstance(y_train, (np.ndarray, pd.DataFrame)):
            self.y_min = np.min(y_train)
            self.y_max = np.max(y_train)

        if isinstance(X_train, torch.Tensor):
            # todo: bad idea, switching memory
            feature_mins, _ = X_train.min(dim=0)
            feature_maxs, _ = X_train.max(dim=0)

            feature_mins = from_tensor_to_numpy(feature_mins).astype(np.float32)
            feature_maxs = from_tensor_to_numpy(feature_maxs).astype(np.float32)

            self.feature_maxs = pd.Series(feature_maxs, index=self.columns)
            self.feature_mins = pd.Series(feature_mins, index=self.columns)
        if isinstance(y_train, torch.Tensor):
            self.y_min = from_tensor_to_numpy(y_train.min())
            self.y_max = from_tensor_to_numpy(y_train.max())

        # print(self.feature_mins, self.feature_maxs)
        # print(self.y_min, self.y_max)

    def set_max_min2(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            # todo: bad idea, switching memory
            feature_mins = preds.min()
            feature_maxs = preds.max()

            self.feature_mins = from_tensor_to_numpy(feature_mins).reshape(-1)
            self.feature_maxs = from_tensor_to_numpy(feature_maxs).reshape(-1)
        if isinstance(targets, torch.Tensor):
            self.y_min = from_tensor_to_numpy(targets.min())
            self.y_max = from_tensor_to_numpy(targets.max())

    @staticmethod
    def cats_to_probs(cats: pd.Series):
        probs = cats.value_counts(normalize=True).to_dict()
        return cats.replace(probs)

    def compute_je_vector(self, X_train, y_train):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.columns)
        if not isinstance(y_train, pd.DataFrame):
            y_train = pd.DataFrame(y_train)

        je_true_list = []

        for col in X_train.columns:
            xi = X_train[col]
            if col in self.discrete_columns:
                x_= self.cats_to_probs(xi).values
            else:
                x_ = xi.values
            je_true_list.append(
                self.numpy_joint_entropy(x_,
                                         y_train.values,
                                         num_bins=self.num_bins,
                                         y_range=[self.y_min, self.y_max],
                                         x_range=[self.feature_mins[col], self.feature_maxs[col]])
            )
        return np.array(je_true_list, dtype=np.float32)

class EntropyMatcherLoss(nn.Module):
    def __init__(self,
                 bins,
                 sigma: float,
                 je_true_vector: np.ndarray = None,
                 lambda_entropy: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.num_bins = int(num_bins)
        self.bins = bins
        self.sigma = float(sigma)
        if not je_true_vector is None:
            self.je_true_vector = torch.tensor(je_true_vector, dtype=torch.float32)
        self.lambda_entropy = float(lambda_entropy)

        # bin_centers shaped to broadcast with (N, D, 1) -> (N, D, B)
        # bin_centers = torch.linspace(0.0, 1.0, self.num_bins).view(1, 1, self.num_bins)
        bin_centers = torch.from_numpy(bins).view(1, 1, -1)

        self.register_buffer("bin_centers", bin_centers)

        inv_two_sigma2 = 1.0 / (2.0 * self.sigma * self.sigma)
        self.register_buffer("inv_two_sigma2", torch.tensor(inv_two_sigma2))

    def _soft_bin(self, values01: torch.Tensor) -> torch.Tensor:
        # values01: (N, D) or (N, 1)
        dist2 = (values01.unsqueeze(2) - self.bin_centers).pow(2)  # (N,D,B)
        weight_logits = -dist2 * self.inv_two_sigma2
        return torch.softmax(weight_logits, dim=2)  # (N, D, B)

    def _normalize01(self, data: torch.Tensor) -> torch.Tensor:
        mx = data.max(dim=0, keepdim=True).values
        mn = data.min(dim=0, keepdim=True).values
        rng = torch.clamp(mx - mn, min=1e-8)
        return torch.clamp((data - mn) / rng, 0.0, 1.0)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, xb: torch.Tensor):
        """
        preds: (N,1)
        targets: (N,1)
        xb: (N,D)
        """
        mse_loss = self.mse(preds, targets)
        preds01 = torch.sigmoid(preds)  # (N,1)

        # preds01 = self._normalize01(preds)
        targets01 = self._normalize01(targets)
        # Normalize features (per-column)
        # x01 = self._normalize01(xb)     # (N,D)

        # Soft bin weights
        # wx = self._soft_bin(x01)        # (N,D,B)
        wx = self._soft_bin(targets01)
        # print(wx)
        wy = self._soft_bin(preds01)    # (N,1,B)
        # print(wy)
        # Squeeze prediction bin axis to (N,B)
        wy_s = wy.squeeze(1)            # (N,B)

        alpha = 1e-3

        # Correct einsum: distinct labels for the two bin axes (k and l).
        # wx: (N, D, B) -> 'ndk'
        # wy_s: (N, B)   -> 'nl'
        # result: (D, k, l) == (D, B, B)
        counts = torch.einsum('ndk,nl->dkl', wx, wy_s) + alpha  # (D,B,B)

        prob = counts / (counts.sum(dim=(1, 2), keepdim=True) + 1e-8)  # (D,B,B)
        log_p = torch.log2(prob + 1e-8)
        je_vec = -(prob * log_p).sum(dim=(1, 2))  # (D,)

        # Optional sanity check: je_true_vector length matches number of features
        # D = xb.shape[1]
        # if self.je_true_vector.numel() != D:
        #     raise ValueError(f"je_true_vector length ({self.je_true_vector.numel()}) != number of features D ({D})")

        # je_true = self.je_true_vector.to(je_vec.device).view(-1)
        # print(torch.stack([je_true, je_vec]))
        # ent_loss = torch.linalg.vector_norm(je_vec - je_true)

        return 0 * mse_loss + self.lambda_entropy * je_vec, from_tensor_to_numpy(je_vec)

class RegressionTrainer:
    def __init__(self, model: nn.Module, loss_mod: nn.Module, ref_model: EntropyPrecomputer, lr: float = 1e-3):
        self.model = model
        self.loss_mod = loss_mod
        self.ref_mod = ref_model
        self.lr = lr
        self.grad_history = {name: [] for name, p in model.named_parameters() if p.requires_grad}
        self.total_grad_norm_history = []
        self.entr_true = []
        self.entr_pred = []

    def make_dataloader(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> DataLoader:
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10, batch_size: int = 256):
        losses = []
        device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device_local)
        self.loss_mod.to(device_local)
        train_dl = self.make_dataloader(X, y, batch_size=batch_size)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bar = tqdm(range(1, epochs + 1), disable=False)
        for epoch in bar:
            grad_sums = {name: 0.0 for name in self.grad_history.keys()}
            total_grad_norm_sum = 0.0
            grad_count = 0
            self.model.train()
            for step, (xb, yb) in enumerate(train_dl, start=1):
                local_entr_true = []
                local_entr_pred = []

                xb = xb.to(device_local, non_blocking=True)
                yb = yb.to(device_local, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                preds = self.model(xb)
                loss, entr_est = self.loss_mod(preds, yb, xb)

                # print(self.ref_mod.feature_ranges)
                # self.ref_mod.set_max_min(xb, preds)
                self.ref_mod.set_max_min2(yb, preds)
                # print(self.ref_mod.feature_ranges)
                # ref_vals = self.ref_mod.compute_je_vector(
                #     from_tensor_to_numpy(yb), from_tensor_to_numpy(preds)
                # )
                ref_vals = self.ref_mod.numpy_joint_entropy(preds, yb)

                local_entr_true.append(np.mean(ref_vals))
                local_entr_pred.append(np.mean(entr_est[0]))

                loss.backward()
                per_param_norms = []
                for name, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    gnorm = p.grad.detach().data.norm(2).item()
                    grad_sums[name] += gnorm
                    per_param_norms.append(gnorm)
                if per_param_norms:
                    total_grad_norm_sum += float(np.linalg.norm(per_param_norms, ord=2))
                grad_count += 1
                opt.step()
                bar.set_description(f"epoch {epoch}/{epochs}, loss: {loss.item():.4f}")

            self.entr_pred.append(np.mean(local_entr_pred))
            self.entr_true.append(np.mean(local_entr_true))
            losses.append(loss.mean().item())
            if grad_count > 0:
                for name in self.grad_history.keys():
                    self.grad_history[name].append(grad_sums[name] / grad_count)
                self.total_grad_norm_history.append(total_grad_norm_sum / grad_count)
        if not os.path.exists("test_Loss_reg"):
            make_dot(loss, params=dict(self.model.named_parameters())).render("test_Loss_reg", format="svg")
        return losses

if __name__ == "__main__":
    # num_bins = 30
    sigma = 1
    lambda_entropy = 1

    epochs = 500
    batch_size = 64
    target = "Delivery_Time_min"

    df = pd.read_csv("Food_Delivery_Times.csv").dropna().drop(columns=["Order_ID"])
    print(df.shape)
    # print(df.columns)

    disc_cols = df.select_dtypes(include=["object"]).columns
    df[disc_cols] = df[disc_cols].apply(LabelEncoder().fit_transform)
    df[[target, "Preparation_Time_min"]] = df[[target, "Preparation_Time_min"]].astype("float")

    y = pd.DataFrame(df.pop(target))

    cols = df.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3)
    from sklearn.preprocessing import MinMaxScaler
    bins = numpy.histogram_bin_edges(MinMaxScaler().fit_transform(y_train))

    print(bins.shape)
    X_model = torch.from_numpy(x_train.values).float()

    model = NeuralNetwork(feature_num=X_model.shape[1])

    # pre = EntropyPrecomputer(num_bins=num_bins, sigma=sigma)
    # pre.columns = cols
    # pre.discrete_columns = disc_cols
    # pre.set_max_min(x_train, y_train)
    # je_true_vector = pre.compute_je_vector(df, y)
    # del pre

    ref_mod = EntropyPrecomputer(bins=bins, sigma=sigma)
    ref_mod.columns = cols

    loss_mod = EntropyMatcherLoss(bins=bins, sigma=sigma,
                                  # je_true_vector=je_true_vector,
                                  lambda_entropy=lambda_entropy)

    trainer = RegressionTrainer(model, loss_mod, lr=1e-3, ref_model=ref_mod)

    y_model = torch.from_numpy(y_train.values).reshape(-1, 1).float()
    losses = trainer.train(X_model, y_model, epochs=epochs, batch_size=batch_size)

    preds = (
        model(torch.from_numpy(x_test.values).to(device).float())
        .float()
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    y_true = y_test.values.astype(np.float32).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))

    # print(np.min(entrop_loss), np.max(entrop_loss), np.mean(entrop_loss), np.std(entrop_loss))
    # plt.plot(losses, label="L")
    plt.title(f"Train Loss (MSE + λ·JE_match). RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    plt.plot(losses, label="entr_L")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.ylim([0, 1000])
    plt.grid()
    plt.legend()
    plt.show()

    from scipy.signal import savgol_filter
    plt.title("Diff b/w entropy estimated and entropy from numpy")
    plt.plot(savgol_filter(
        np.array(trainer.entr_pred) - np.array(trainer.entr_true),
        window_length=20, polyorder=2),
             label="Entropy diff smoothed")
    plt.plot(np.array(trainer.entr_pred) - np.array(trainer.entr_true),
             alpha=0.5,
             label="Entropy diff true")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(trainer.entr_pred, label="entr_pred")
    plt.plot(trainer.entr_true, label="entr_true")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.grid()
    plt.legend()
    plt.show()

    # plt.scatter(y_true, preds, s=10, alpha=0.6)
    # plt.xlabel("y_true")
    # plt.ylabel("y_pred")
    # plt.title("Predictions vs True")
    # lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    # plt.plot(lims, lims, 'r--', linewidth=1)
    # plt.grid()
    # plt.show()

    plt.plot(trainer.total_grad_norm_history, label="total_grad_norm")
    plt.title("Total gradient L2 norm per epoch")
    plt.xlabel("epoch")
    plt.ylabel("grad norm")
    plt.grid()
    plt.legend()
    plt.show()
