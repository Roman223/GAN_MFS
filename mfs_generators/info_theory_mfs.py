import torch

class CategoricalMFS:
    device = torch.device("cpu")

    def change_device(self, device):
        self.device = device

    @staticmethod
    def get_cpt(row: torch.Tensor, col: torch.Tensor, normalize="all", num_rows=None, num_cols=None):
        assert row.ndim == 1 and col.ndim == 1, "row and col must be 1D tensors"
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


    @staticmethod
    def ft_joint_entropy(crosstab, epsilon: float = 1e-8) -> torch.Tensor:
        joint_prob = crosstab + epsilon
        joint_entropy = -torch.sum(joint_prob * torch.log2(joint_prob))
        return joint_entropy

    def get_mfs(self, X, y, subset=None):
        if subset is None:
            subset = ["mean", "var"]

        mfs = []
        for name in subset:
            if name not in self.feature_methods:
                raise ValueError(f"Unsupported meta-feature: '{name}'")

                if y is None:
                    raise ValueError("Meta-feature 'gravity' requires `y`.")
                res = self.feature_methods[name](X, y)
                res = torch.tile(res, (X.shape[-1],))  # match dimensionality

            mfs.append(res)
        shapes = [i.shape.numel() for i in mfs]
        mfs = [self.pad_only(mf, max(shapes)) for mf in mfs]
        return torch.stack(mfs)