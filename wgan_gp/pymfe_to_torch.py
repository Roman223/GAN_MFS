import torch
from typing import Optional, Union

from pymfe.mfe import MFE
import pandas as pd
import numpy as np

class MFEToTorch:
    """
    A class to compute meta-features using PyTorch.
    
        This class provides methods to calculate various meta-features for a given
        dataset using PyTorch tensors. It includes functionalities for computing
        statistical measures, correlation, covariance, and other properties of the
        data.
    
        Class Attributes:
        - device
    
        Class Methods:
        - feature_methods:
    """

    device = torch.device("cpu")

    @property
    def feature_methods(self):
        """
        Returns a dictionary that maps feature names to their corresponding extraction methods.
        
                This mapping is essential for calculating a comprehensive set of statistical
                properties on both real and synthetic datasets. These features are then
                used to evaluate the quality and utility of the generated synthetic data
                by comparing them against the features of the real data.
        
                Returns:
                    dict: A dictionary where keys are feature names (strings) and
                        values are the corresponding feature extraction methods.
                        The dictionary includes the following features:
                        'cor' (correlation), 'cov' (covariance), 'eigenvalues',
                        'iq_range' (interquartile range), 'gravity', 'kurtosis',
                        'skewness', 'mad' (median absolute deviation), 'max', 'min',
                        'mean', 'median', 'range', 'sd' (standard deviation),
                        'var' (variance), and 'sparsity'.
        """
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
        """
        Computes the gravity between the majority and minority classes.
        
                This method calculates the distance between the mean feature vectors of the
                majority and minority classes. This distance serves as a measure of class
                separation in the feature space. By computing this "gravity," the method
                quantifies the dissimilarity between the most and least frequent classes,
                providing insight into the dataset's class distribution and feature
                representation. This information can be valuable for assessing the quality
                and representativeness of generated synthetic data compared to real data.
                
                Args:
                    N: Feature tensor of shape (num_instances, num_features).
                    y: Target tensor of shape (num_instances,).
                    norm_ord: Order of the norm to compute the distance (e.g., 2 for Euclidean). Defaults to 2.
                    classes: Optional tensor of unique class labels. If None, it's computed from `y`.
                    class_freqs: Optional tensor of class frequencies. If None, it's computed from `y`.
                    cls_inds: Optional list of indices for each class. If provided, it uses these indices to select instances.
                
                Returns:
                    torch.Tensor: The gravity value, representing the distance between the class centers.
        """
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
        """
        Changes the device where computations will be performed.
        
        Args:
            device (str): The target device (e.g., 'cpu', 'cuda').
        
        Returns:
            None
        
        This method is crucial for ensuring that the model and data reside on the same device,
        allowing for efficient computation and utilization of available hardware resources
        during the synthetic data generation and evaluation processes.
        """
        self.device = device

    @staticmethod
    def cov(tensor, rowvar=True, bias=False):
        """
        Estimates the covariance matrix of a given tensor, crucial for understanding the statistical relationships within the data. This is a key step in evaluating how well the generated synthetic data captures the underlying dependencies present in the original data.
        
                Args:
                    tensor (torch.Tensor): Input data tensor.
                    rowvar (bool, optional): If True (default), rows represent variables, with observations in the columns. If False, columns represent variables.
                    bias (bool, optional): If False (default), then the normalization is by N-1. Otherwise, normalization is by N.
        
                Returns:
                    torch.Tensor: The covariance matrix of the input tensor.
        """
        tensor = tensor if rowvar else tensor.transpose(-1, -2)
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
        return factor * tensor @ tensor.transpose(-1, -2).conj()

    def corrcoef(self, tensor, rowvar=True):
        """
        Calculates the Pearson product-moment correlation coefficients, normalizing the covariance matrix by the standard deviations to obtain correlation values. This provides a measure of the linear relationship between variables in the input tensor, which is useful for comparing real and synthetic data.
        
                Args:
                    tensor (torch.Tensor): Input data tensor.
                    rowvar (bool, optional): If True (default), rows represent variables, with observations in the columns. Otherwise, columns represent variables.
        
                Returns:
                    torch.Tensor: Pearson product-moment correlation coefficients matrix.
        """
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
        """
        Calculates the absolute values of the lower triangle elements of a correlation matrix to quantify feature dependencies.
        
                This method computes the correlation matrix of the input tensor `N`,
                extracts the elements from the lower triangle (excluding the diagonal),
                and returns the absolute values of these elements. This is done to summarize the relationships between features,
                which is useful for evaluating how well the synthetic data captures the dependencies present in the real data.
                By focusing on the lower triangle and taking absolute values, the method efficiently provides a measure of feature interconnectedness,
                ignoring self-correlations and directionality.
        
                Args:
                    N: The input tensor for which to compute the correlation matrix.
        
                Returns:
                    torch.Tensor: A tensor containing the absolute values of the elements
                        in the lower triangle of the correlation matrix.
        """
        corr_mat = self.corrcoef(N, rowvar=False)
        res_num_rows, _ = corr_mat.shape

        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = corr_mat[tril_indices[0], tril_indices[1]]

        return torch.abs(inf_triang_vals)

    def ft_cov_torch(
            self,
            N: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the absolute values of the lower triangular elements of the covariance matrix. This focuses on the relationships between variables, extracting the lower triangle to reduce redundancy and focusing on key covariance values. The absolute value ensures that the magnitude of the covariance is considered, regardless of the direction of the relationship.
        
                Args:
                    N: Input tensor for covariance calculation.
        
                Returns:
                    torch.Tensor: A tensor containing the absolute values of the lower triangular elements of the covariance matrix.
        """
        cov_mat = self.cov(N, rowvar=False)

        res_num_rows = cov_mat.shape[0]
        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = cov_mat[tril_indices[0], tril_indices[1]]

        return torch.abs(inf_triang_vals)

    def ft_eigenvals(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the eigenvalues of the covariance matrix of the input tensor.
        
                This function is crucial for assessing the diversity and information
                content of the input data. By calculating the eigenvalues of the
                covariance matrix, we gain insights into the principal components
                and variance distribution within the data, which helps to ensure
                the generated synthetic data retains the key statistical
                characteristics of the original data.
        
                Args:
                    x (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: The eigenvalues of the covariance matrix.
        """
        # taking real part of first two eigenvals
        centered = x - x.mean(dim=0, keepdim=True)
        covs = self.cov(centered, rowvar=False)
        return torch.linalg.eigvalsh(covs)

    @staticmethod
    def ft_iq_range(X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the interquartile range (IQR) of a tensor along the first dimension.
        
                The IQR is a measure of statistical dispersion, representing the difference between the 75th and 25th percentiles. This is useful for understanding the spread of the data, which helps to assess the utility of generated synthetic data by comparing its distribution to the real data.
        
                Args:
                    X: The input tensor of shape [num_samples, num_features].
        
                Returns:
                    The interquartile range of the input tensor, with shape [num_features]. This represents the spread of each feature across the samples.
        """
        q75, q25 = torch.quantile(X, 0.75, dim=0), torch.quantile(X, 0.25, dim=0)
        iqr = q75 - q25  # shape: [num_features]
        return iqr

    @staticmethod
    def ft_kurtosis(x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the kurtosis of a tensor.
        
                This function computes the kurtosis of the input tensor `x`, a statistical measure
                describing the shape of the data's distribution, specifically its tailedness.
                By calculating kurtosis, we can assess how well the generated data's distribution
                matches that of the real data, ensuring the synthetic data retains similar statistical
                properties. This is crucial for maintaining the utility of the generated data in downstream tasks.
        
                Args:
                    x (torch.Tensor): Input tensor.
        
                Returns:
                    torch.Tensor: The kurtosis of the input tensor.
        """
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return kurtoses

    @staticmethod
    def ft_skewness(x: torch.Tensor) -> torch.Tensor:
        """
        Computes the skewness of a tensor.
        
                This function calculates the skewness of the input tensor, a key statistical
                measure reflecting the asymmetry of the data distribution. Preserving this characteristic
                is crucial when generating synthetic data to maintain the real data's statistical properties.
        
                Args:
                    x (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: The skewness of the input tensor.
        """
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    @staticmethod
    def ft_mad(x: torch.Tensor, factor: float = 1.4826) -> torch.Tensor:
        """
        Compute the Median Absolute Deviation (MAD) of a tensor.
        
                The MAD is a robust measure of statistical dispersion, useful for
                understanding the spread of data in both real and synthetic datasets.
                It helps assess how well the generated data captures the variability
                present in the original data.
        
                Args:
                    x: The input tensor.
                    factor: A scaling factor to make the MAD an unbiased estimator of the
                        standard deviation for normal data. Default is 1.4826, which
                        applies when the data is normally distributed.
        
                Returns:
                    torch.Tensor: The MAD of the input tensor.
        """
        m = x.median(dim=0, keepdim=True).values
        ama = torch.abs(x - m)
        mama = ama.median(dim=0).values
        return mama / (1 / factor)

    @staticmethod
    def ft_mean(N: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of a tensor along the first dimension to aggregate information across samples. This is useful for summarizing the central tendency of features in the generated or real data.
        
                Args:
                    N (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: The mean of the input tensor along dimension 0.
        """
        return N.mean(dim=0)

    @staticmethod
    def ft_max(N: torch.Tensor) -> torch.Tensor:
        """
        Finds the maximum value in a tensor along dimension 0. This is used to identify the most prominent features across a dataset, which is crucial for maintaining data utility in generated synthetic data.
        
                Args:
                    N (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: A tensor containing the maximum values along dimension 0.
        """
        return N.max(dim=0, keepdim=False).values

    @staticmethod
    def ft_median(N: torch.Tensor) -> torch.Tensor:
        """
        Calculates the median of a tensor along the first dimension. This is used to derive a representative central tendency of the data distribution, which is a crucial aspect of maintaining data utility in synthetic data generation.
        
                Args:
                    N: The input tensor.
        
                Returns:
                    torch.Tensor: A tensor containing the median values along the first dimension.
        """
        return N.median(dim=0).values

    @staticmethod
    def ft_min(N: torch.Tensor) -> torch.Tensor:
        """
        Finds the minimum value of a tensor along dimension 0, which is useful for identifying the smallest values across different samples when comparing real and synthetic data distributions.
        
                Args:
                    N (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: A tensor containing the minimum values along dimension 0. This represents the minimum feature values across the dataset, aiding in the comparison of feature ranges between real and synthetic datasets.
        """
        return N.min(dim=0).values

    @staticmethod
    def ft_var(N):
        """
        Calculates the variance of a tensor along dimension 0. This is a crucial step in assessing the statistical similarity between real and synthetic datasets generated by the GAN, ensuring that the generated data captures the variability present in the original data.
        
                Args:
                    N (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: The variance of the input tensor along dimension 0.
        """
        return torch.var(N, dim=0)

    @staticmethod
    def ft_std(N):
        """
        Calculates the standard deviation of a tensor along the first dimension (dimension 0). This is used to understand the spread or dispersion of the generated synthetic data across different samples, ensuring the generated data maintains a similar statistical distribution to the real data.
        
                Args:
                    N (torch.Tensor): The input tensor representing a batch of generated samples.
        
                Returns:
                    torch.Tensor: The standard deviation of the input tensor along dimension 0, representing the standard deviation for each feature across the generated samples.
        """
        return torch.std(N, dim=0)

    @staticmethod
    def ft_range(N: torch.Tensor) -> torch.Tensor:
        """
        Calculates the range of values (max - min) along the first dimension (dimension 0) of the input tensor. This is useful for understanding the spread or variability of the data along that dimension, which helps assess how well the generated data captures the characteristics of the original data.
        
        Args:
            N: The input tensor.
        
        Returns:
            torch.Tensor: A tensor containing the range (max - min) of values along dimension 0.
        """
        return N.max(dim=0).values - N.min(dim=0).values

    def ft_sparsity(self, N: torch.Tensor) -> torch.Tensor:
        """
        Calculates the feature sparsity of a given tensor.
        
                This method computes the sparsity of each feature in the input tensor `N`.
                Sparsity is defined as the ratio of the total number of instances to the
                number of unique values for each feature, normalized to the range [0, 1].
                This metric helps to assess the diversity of feature values, which is crucial
                for generating synthetic data that accurately reflects the statistical
                properties of the original dataset. By quantifying feature sparsity, we can
                ensure that the generated data maintains a similar level of variability
                as the real data, thereby preserving its utility for downstream tasks.
        
                Args:
                    N (torch.Tensor): A tensor of shape (num_instances, num_features) representing the input data.
        
                Returns:
                    torch.Tensor: A tensor of shape (num_features,) containing the sparsity
                    score for each feature, normalized to the range [0, 1]. The tensor is
                    moved to the device specified by `self.device`.
        """
        ans = torch.tensor([attr.size(0) / torch.unique(attr).size(0) for attr in N.T])

        num_inst = N.size(0)
        norm_factor = 1.0 / (num_inst - 1.0)
        result = (ans - 1.0) * norm_factor

        return result.to(self.device)


    def pad_only(self, tensor, target_len):
        """
        Pads a tensor with zeros to a specified length, ensuring consistent input sizes for subsequent processing steps. This is particularly useful when dealing with variable-length sequences that need to be batched or processed by models requiring fixed-size inputs.
        
                Args:
                    tensor (torch.Tensor): The input tensor to be padded.
                    target_len (int): The desired length of the padded tensor.
        
                Returns:
                    torch.Tensor: The padded tensor, or the original tensor if its length is already greater than or equal to `target_len`.
        """
        if tensor.shape[0] < target_len:
            padding = torch.zeros(target_len - tensor.shape[0]).to(self.device)
            return torch.cat([tensor, padding])

        return tensor

    def get_mfs(self, X, y, subset=None):
        """
        Computes a set of meta-features on the input data. These meta-features capture essential characteristics of the dataset, which is crucial for evaluating and ensuring the utility of synthetic data generated by GANs.
        
                Args:
                    X (torch.Tensor): The input data tensor.
                    y (torch.Tensor, optional): The target variable tensor. Required if 'gravity' is in the subset.
                    subset (list of str, optional): A list of meta-feature names to compute. If None, defaults to ['mean', 'var'].
        
                Returns:
                    torch.Tensor: A tensor containing the computed meta-features, padded to the maximum shape among the computed features and stacked into a single tensor. This allows for consistent representation and comparison of different meta-features.
        """
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
        """
        Compares meta-feature extraction using the `pymfe` package and the `MFEToTorch` class.
        
            This method fetches the California Housing dataset, extracts meta-features using both `pymfe` and the `MFEToTorch` class, and then compares the results. This comparison helps validate the correctness and consistency of the meta-feature extraction process implemented in the `MFEToTorch` class, ensuring that it aligns with established meta-feature extraction tools.
        
            Args:
                subset (list, optional): A list of meta-features to extract. If None, defaults to ["mean", "var"].
        
            Returns:
                pandas.DataFrame: A DataFrame containing the meta-features extracted by both `pymfe` and `MFEToTorch`, along with any discrepancies between the two.
        """
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

# MFEToTorch().test_me(subset=["mean", "var", "eigenvalues", "gravity"])