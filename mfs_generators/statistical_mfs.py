import torch
from typing import Optional, Union

from pymfe.mfe import MFE
import pandas as pd
import numpy as np

class StatisticalMFS:
    device = torch.device("cpu")

    @property
    def feature_methods(self):
        return {
            "cor": self.ft_cor_torch,
            "cov": self.ft_cov_torch,
            "eigenvalues": self.ft_eigenvals,
            "iq_range": self.ft_iq_range,
            "gravity": self.ft_gravity_torch,
            "kurtosis": self.ft_kurtosis,
            "skewness": self.ft_skewness,
            "mad": self.ft_mad,
            "max": self.ft_max,
            "min": self.ft_min,
            "mean": self.ft_mean,
            "median": self.ft_median,
            "range": self.ft_range,
            "sd": self.ft_std,
            "var": self.ft_var,
            "sparsity": self.ft_sparsity,
        }

    @staticmethod
    def ft_gravity_torch(
            N: torch.Tensor,
            y: torch.Tensor,
            norm_ord: Union[int, float] = 2,
            classes: Optional[torch.Tensor] = None,
            class_freqs: Optional[torch.Tensor] = None,
            cls_inds: Optional[torch.Tensor] = None,
    ):
        if classes is None or class_freqs is None:
            classes, class_freqs = torch.unique(y, return_counts=True)

        ind_cls_maj = torch.argmax(class_freqs)
        class_maj = classes[ind_cls_maj]

        remaining_classes = torch.cat((classes[:ind_cls_maj], classes[ind_cls_maj + 1:]))
        remaining_freqs = torch.cat((class_freqs[:ind_cls_maj], class_freqs[ind_cls_maj + 1:]))

        ind_cls_min = torch.argmin(remaining_freqs)

        if cls_inds is not None:
            insts_cls_maj = N[cls_inds[ind_cls_maj]]
            if ind_cls_min >= ind_cls_maj:
                ind_cls_min += 1
            insts_cls_min = N[cls_inds[ind_cls_min]]
        else:
            class_min = remaining_classes[ind_cls_min]
            insts_cls_maj = N[y == class_maj]
            insts_cls_min = N[y == class_min]

        center_maj = insts_cls_maj.mean(dim=0)
        center_min = insts_cls_min.mean(dim=0)
        gravity = torch.norm(center_maj - center_min, p=norm_ord)

        return gravity

    def change_device(self, device):
        self.device = device

    @staticmethod
    def cov(tensor, rowvar=True, bias=False):
        """Estimate a covariance matrix (np.cov)"""
        tensor = tensor if rowvar else tensor.transpose(-1, -2)
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
        return factor * tensor @ tensor.transpose(-1, -2).conj()

    def corrcoef(self, tensor, rowvar=True):
        """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
        covariance = self.cov(tensor, rowvar=rowvar)
        variance = covariance.diagonal(0, -1, -2)
        if variance.is_complex():
            variance = variance.real
        stddev = variance.sqrt()
        covariance /= stddev.unsqueeze(-1)
        covariance /= stddev.unsqueeze(-2)
        if covariance.is_complex():
            covariance.real.clip_(-1, 1)
            covariance.imag.clip_(-1, 1)
        else:
            covariance.clip_(-1, 1)
        return covariance

    def ft_cor_torch(self, N: torch.Tensor) -> torch.Tensor:
        corr_mat = self.corrcoef(N, rowvar=False)
        res_num_rows, _ = corr_mat.shape

        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = corr_mat[tril_indices[0], tril_indices[1]]

        return torch.abs(inf_triang_vals)

    def ft_cov_torch(
            self,
            N: torch.Tensor,
    ) -> torch.Tensor:
        cov_mat = self.cov(N, rowvar=False)

        res_num_rows = cov_mat.shape[0]
        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = cov_mat[tril_indices[0], tril_indices[1]]

        return torch.abs(inf_triang_vals)

    def ft_eigenvals(self, x: torch.Tensor) -> torch.Tensor:
        # taking real part of first two eigenvals
        centered = x - x.mean(dim=0, keepdim=True)
        covs = self.cov(centered, rowvar=False)
        return torch.linalg.eigvalsh(covs)

    @staticmethod
    def ft_iq_range(X: torch.Tensor) -> torch.Tensor:
        q75, q25 = torch.quantile(X, 0.75, dim=0), torch.quantile(X, 0.25, dim=0)
        iqr = q75 - q25  # shape: [num_features]
        return iqr

    @staticmethod
    def ft_kurtosis(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return kurtoses

    @staticmethod
    def ft_skewness(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    @staticmethod
    def ft_mad(x: torch.Tensor, factor: float = 1.4826) -> torch.Tensor:
        m = x.median(dim=0, keepdim=True).values
        ama = torch.abs(x - m)
        mama = ama.median(dim=0).values
        return mama / (1 / factor)

    @staticmethod
    def ft_mean(N: torch.Tensor) -> torch.Tensor:
        return N.mean(dim=0)

    @staticmethod
    def ft_max(N: torch.Tensor) -> torch.Tensor:
        return N.max(dim=0, keepdim=False).values

    @staticmethod
    def ft_median(N: torch.Tensor) -> torch.Tensor:
        return N.median(dim=0).values

    @staticmethod
    def ft_min(N: torch.Tensor) -> torch.Tensor:
        return N.min(dim=0).values

    @staticmethod
    def ft_var(N):
        return torch.var(N, dim=0)

    @staticmethod
    def ft_std(N):
        return torch.std(N, dim=0)

    @staticmethod
    def ft_range(N: torch.Tensor) -> torch.Tensor:
        return N.max(dim=0).values - N.min(dim=0).values

    def ft_sparsity(self, N: torch.Tensor) -> torch.Tensor:
        ans = torch.tensor([attr.size(0) / torch.unique(attr).size(0) for attr in N.T])

        num_inst = N.size(0)
        norm_factor = 1.0 / (num_inst - 1.0)
        result = (ans - 1.0) * norm_factor

        return result.to(self.device)


    def pad_only(self, tensor, target_len):
        if tensor.shape[0] < target_len:
            padding = torch.zeros(target_len - tensor.shape[0]).to(self.device)
            return torch.cat([tensor, padding])

        return tensor

    def get_mfs(self, X, y, subset=None):
        if subset is None:
            subset = ["mean", "var"]

        mfs = []
        for name in subset:
            if name not in self.feature_methods:
                raise ValueError(f"Unsupported meta-feature: '{name}'")

            if name == "gravity":
                if y is None:
                    raise ValueError("Meta-feature 'gravity' requires `y`.")
                res = self.feature_methods[name](X, y)
                res = torch.tile(res, (X.shape[-1],))  # match dimensionality
            else:
                res = self.feature_methods[name](X)

            mfs.append(res)
        shapes = [i.shape.numel() for i in mfs]
        mfs = [self.pad_only(mf, max(shapes)) for mf in mfs]
        return torch.stack(mfs)

    def test_me(self, subset=None):
        """Can be outdated"""
        if subset is None:
            subset = ["mean", "var"]

        from sklearn.datasets import fetch_california_housing
        bunch = fetch_california_housing(as_frame=True)
        X, y = bunch.data, bunch.target
        print(f"Init data shape: {X.shape} + {y.shape}")

        mfe = MFE(groups="statistical", summary=None)
        mfe.fit(X.values, y.values)
        ft = mfe.extract()

        pymfe = pd.DataFrame(
            map(lambda x: [x], ft[1]), index=ft[0], columns=["pymfe"]).dropna()

        X_tensor = torch.tensor(X.values)
        y_tensor = torch.tensor(y)

        mfs = self.get_mfs(X_tensor, y_tensor, subset).numpy()
        mfs_df = pd.DataFrame({'torch_mfs': list(mfs)})

        mfs_df.index = subset
        # mfs_df = mfs_df.reindex(self.mfs_available)

        res = pymfe.merge(mfs_df, left_index=True, right_index=True, how="outer")


        def round_element(val, decimals=2):
            if isinstance(val, list):
                return [round(x, decimals) for x in val]
            elif isinstance(val, np.ndarray):
                return np.round(val, decimals)
            return round(val, decimals)

        res = res.map(lambda x: round_element(x, 5)).dropna()

        print(res)

# StatisticalMFS().test_me(subset=["gravity"])