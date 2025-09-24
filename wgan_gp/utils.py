from scipy.spatial.distance import jensenshannon, cosine

from ripser import ripser
from persim import wasserstein
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

def sample_from_Xy_tensors(X:torch.Tensor, n_samples):
    """
    Samples a subset of rows from a given data tensor to create a smaller, representative dataset. This is useful for efficiently training and evaluating models on a manageable portion of the data while preserving its key characteristics.
    
        Args:
            X (torch.Tensor): The input data tensor to sample from.
            n_samples (int): The number of samples to extract.
    
        Returns:
            torch.Tensor: A tensor containing the sampled rows from the input tensor X, representing a subset of the original data.
    """
    indices = torch.randperm(X.size(0))
    indices_trunc = indices[:n_samples]
    X_sampled_tensor = X[indices_trunc[:n_samples]]
    # y_sampled_tensor = y[indices_trunc[:n_samples]]
    # return X_sampled_tensor, y_sampled_tensor
    return X_sampled_tensor

def random_sample(arr: np.array, size: int = 1) -> np.array:
    """
    Generates a random sample from an array without replacement. This is useful for creating subsets of the original data to train and evaluate machine learning models, ensuring that the evaluation is performed on unseen data.
    
        Args:
            arr (np.array): The input array.
            size (int): The number of samples to generate. Defaults to 1.
    
        Returns:
            np.array: A new array containing the random sample.
    """
    return arr[np.random.choice(len(arr), size=size, replace=False)]

def convert_result(results):
    """
    Converts the model results into a flattened dictionary format suitable for evaluating the performance of synthetic data generation.
    
        This method takes a dictionary of model results, calculates the mean and
        standard deviation of the metrics for each model, and then flattens the
        results into a single row dictionary. This flattened structure facilitates
        comparison and analysis of different synthetic data generation approaches
        by providing a consolidated view of their performance metrics.
    
        Args:
          results: A dictionary where keys are model names and values are
            dictionaries of metrics (e.g., mape, rmse, r2).
    
        Returns:
          A list containing a single dictionary. The dictionary's keys are
          constructed by concatenating the model name and metric name with mean/std
          (e.g., 'model1_mape_mean', 'model1_rmse_std'), and the values are the
          corresponding calculated values. This format is used to create a single-row
          representation of the aggregated metrics, which is useful for comparing
          different synthetic data generation models.
    """
    df = pd.DataFrame([
        {
            'model': model,
            'mape': metrics['mape'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        }
        for model, metrics in results.items()
    ])
    df_agg = df.groupby("model").agg(["mean", "std"])
    df_agg.columns = ['_'.join(col) for col in df_agg.columns]

    df_flat = df_agg.stack().rename_axis(['model', 'metric']).reset_index()
    df_flat['colname'] = df_flat['model'] + '_' + df_flat['metric']
    df_single_row = df_flat.set_index('colname')[0].to_frame().T
    return df_single_row.to_dict(orient='records')

def calc_utility_metrics(
        synth: torch.Tensor | np.ndarray,
        x_train: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        y_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,):
    """
    Calculates utility metrics to assess how well synthetic data preserves the characteristics of real data for regression tasks.
    
        This function trains a regression model on both real and synthetic datasets and then evaluates the performance of both models on a common test set.
        The comparison of these performances indicates the utility of the synthetic data for maintaining the performance of machine learning models.
        
        Args:
            synth (torch.Tensor | np.ndarray): The synthetic data, where the last column is assumed to be the target variable.
            x_train (torch.Tensor | np.ndarray): The training data features from the real dataset.
            x_test (torch.Tensor | np.ndarray): The test data features from the real dataset.
            y_test (torch.Tensor | np.ndarray): The test data labels from the real dataset.
            y_train (torch.Tensor | np.ndarray): The training data labels from the real dataset.
        
        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries. The first dictionary contains the regression performance metrics achieved using real data,
            and the second dictionary contains the regression performance metrics achieved using synthetic data.
    """

    if isinstance(synth, np.ndarray):
        synth = torch.from_numpy(synth)

    if isinstance(x_train, np.ndarray):
        x_train = torch.from_numpy(x_train)

    if isinstance(x_test, np.ndarray):
        x_test = torch.from_numpy(x_test)

    if isinstance(y_test, np.ndarray):
        y_test = torch.from_numpy(y_test)

    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)

    metrics_real = compute_regression_performance(x_train,
                                                 y_train,
                                                 x_test,
                                                 y_test,)

    synth_X, synth_y = synth[:, :-1], synth[:, -1]

    metrics_synth = compute_regression_performance(synth_X,
                                                   synth_y,
                                                   x_test,
                                                   y_test,)
    return convert_result(metrics_real), convert_result(metrics_synth)


def remove_inf(diagram, replacement=None):
    """
    Replaces infinite death times in a persistence diagram with a finite value to ensure compatibility with downstream analysis and visualization tools. This is necessary because many tools cannot handle infinite values, which can arise in persistence diagrams when topological features persist indefinitely.
    
        Args:
            diagram (np.ndarray): A persistence diagram represented as a NumPy array, where each row corresponds to a topological feature and contains birth and death times.
            replacement (float, optional): The value to use as a replacement for infinite death times. If None, it defaults to 1.1 times the maximum finite death time in the diagram.
    
        Returns:
            np.ndarray: A copy of the input persistence diagram with infinite death times replaced by a finite value.
    """
    if len(diagram) == 0:
        return diagram
    finite_deaths = diagram[np.isfinite(diagram[:, 1]), 1]
    if len(finite_deaths) == 0:
        # All death times are inf — choose arbitrary value
        finite_max = 1.0
    else:
        finite_max = np.max(finite_deaths)
    if replacement is None:
        replacement = 1.1 * finite_max
    diagram = diagram.copy()
    diagram[np.isinf(diagram[:, 1]), 1] = replacement
    return diagram

def topological_distance(X, Y, maxdim=2):
    """
    Compute the topological distance between two point clouds using persistent homology.
    
        This method leverages persistent homology and the Wasserstein distance to
        quantify the dissimilarity between the topological features of two point clouds,
        specifically focusing on 0-dimensional (H0) and 1-dimensional (H1) features.
        By comparing the persistence diagrams, it provides a measure of how different
        the underlying shapes and connectivity patterns are between the datasets.
        This is crucial for evaluating how well the generated data captures the
        essential topological characteristics of the real data.
    
        Args:
            X (numpy.ndarray): The first point cloud.
            Y (numpy.ndarray): The second point cloud.
            maxdim (int, optional): The maximum dimension to compute persistent homology for.
                Defaults to 2.
    
        Returns:
            tuple: A tuple containing the Wasserstein distances for H0 and H1.
            The first element is the Wasserstein distance between the H0
            persistence diagrams. The second element is the Wasserstein
            distance between the H1 persistence diagrams, or None if the
            persistence diagrams do not contain H1.
    """
    dgms_X = ripser(X, maxdim=maxdim)['dgms']
    dgms_Y = ripser(Y, maxdim=maxdim)['dgms']
    h0 = wasserstein(dgms_X[0], dgms_Y[0])
    h1 = wasserstein(dgms_X[1], dgms_Y[1]) if len(dgms_X) > 1 else None
    return h0, h1

def correlation_matrix_distance(df1: pd.DataFrame,
                                df2: pd.DataFrame,
                                metric: str = 'frobenius') -> float:
    """
    Compute the distance between the correlation structures of two datasets.
    
        This function quantifies the dissimilarity in the relationships between variables
        in two datasets by comparing their correlation matrices. This is useful for
        assessing how well synthetic data replicates the correlational behavior of real data.
    
        Args:
            df1 (pandas.DataFrame): The first DataFrame.
            df2 (pandas.DataFrame): The second DataFrame.
            metric (str, optional): The distance metric to use.
                Options are 'frobenius', 'euclidean', 'cosine', and 'spectral'.
                Defaults to 'frobenius'.
    
        Returns:
            float: The distance between the correlation matrices of the two DataFrames.
    """
    # Ensure same columns and order
    common_cols = df1.columns.intersection(df2.columns)
    df1 = df1[common_cols].dropna()
    df2 = df2[common_cols].dropna()

    # Truncate to equal length if needed
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    # Compute correlation matrices
    corr1 = df1.corr().values
    corr2 = df2.corr().values

    # Distance computation
    if metric == 'frobenius':
        return np.linalg.norm(corr1 - corr2, ord='fro')
    elif metric == 'euclidean':
        return np.linalg.norm((corr1 - corr2).ravel())
    elif metric == 'cosine':
        return cosine(corr1.ravel(), corr2.ravel())
    elif metric == 'spectral':
        return np.linalg.norm(np.linalg.eigvalsh(corr1 - corr2), ord=2)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: 'frobenius', 'euclidean', 'cosine', 'spectral'.")

def estimate_marginal_js(df1, df2, epsilon=1e-10):
    """
    Estimates the marginal Jensen-Shannon divergence between two dataframes for common columns.
    
        This function quantifies the dissimilarity between the distributions of individual columns
        in two dataframes. It computes histograms for each common column, normalizes them into
        probability distributions, and then calculates the Jensen-Shannon divergence between
        these distributions. This helps assess how well the synthetic data (df2) replicates
        the distribution of features in the real data (df1).
    
        Args:
            df1: The first dataframe, representing the real data.
            df2: The second dataframe, representing the synthetic data.
            epsilon: A small value added to histogram counts to avoid zero-probabilities (default: 1e-10).
    
        Returns:
            dict: A dictionary where keys are the common column names and values are the
                corresponding Jensen-Shannon divergence scores. Lower scores indicate greater similarity
                between the distributions of the real and synthetic data for that column.
    """
    js_results = {}

    common_cols = df1.columns.intersection(df2.columns)

    for col in common_cols:
        x = df1[col].dropna().values
        y = df2[col].dropna().values

        # Histogram counts (not density, we'll normalize manually)
        p_hist, _ = np.histogram(x)
        q_hist, _ = np.histogram(y)

        # Add small epsilon to avoid zero-probabilities
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon

        # Normalize to get valid distributions
        p_prob = p_hist / p_hist.sum()
        q_prob = q_hist / q_hist.sum()

        # JS divergence is the square of the JS distance from scipy
        js_div = jensenshannon(p_prob, q_prob, base=2) ** 2
        js_results[col] = js_div

    return js_results


def calc_metrics(synth, test):
    """
    Calculates the cosine distance between the correlation matrices of a synthetic and a test dataset. This helps to quantify how well the synthetic data preserves the relationships between features present in the original data. Preserving these relationships is crucial for maintaining the utility of the synthetic data for downstream tasks.
    
        Args:
            synth: The synthetic dataset.
            test: The test dataset (typically the real data).
    
        Returns:
            dict: A dictionary containing the cosine distance between the correlation
                matrices of the synthetic and test datasets, keyed by
                "cosine_dist_corr_matrix". A lower distance indicates better preservation
                of feature relationships.
    """
    return {"cosine_dist_corr_matrix": correlation_matrix_distance(synth, test, metric="cosine")}

def compute_regression_performance(X,
                                   y,
                                   X_test,
                                   y_test):
    """
    Computes and compares the performance of regression models on real data.
    
    This method trains a set of regressors on the provided training data and evaluates their performance on a separate test set. It calculates Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and R-squared (R2) to quantify the accuracy and goodness-of-fit of each regressor. This is done to establish a baseline performance of the models on the real data, which can then be compared with the performance of models trained on synthetic data.
    
    Args:
        X: Training data features.
        y: Training data target.
        X_test: Test data features.
        y_test: Test data target.
    
    Returns:
        dict: A dictionary containing the performance metrics for each regressor.
            The keys of the dictionary are the names of the regressors, and the
            values are dictionaries containing the MAPE, RMSE, and R2 scores.
            For example:
            {'XGBRegressor': {'mape': 0.123, 'rmse': 4.56, 'r2': 0.789}}
    """
    regressors = [XGBRegressor()]
    result = {i.__class__.__name__: {} for i in regressors}

    for regressor in regressors:
        regressor.fit(X, y)
        predictions = regressor.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        local_result = {
            "mape": mape,
            "rmse": rmse,
            "r2": r2,
        }
        result[regressor.__class__.__name__] = local_result
    return result

def create_variates(X: torch.Tensor, y: torch.Tensor=None, sample_frac=0.3, sample_number=100):
    """
    Creates a set of data samples (variates) from the input tensor `X`.
    
        This function generates multiple subsets (variates) of the original data `X`.
        These variates are created by randomly sampling a fraction of the original
        data points.  Creating these diverse samples is crucial for downstream tasks
        that benefit from ensemble methods or require multiple perspectives on the data.
    
        Args:
            X (torch.Tensor): The input tensor representing the original dataset.
            y (torch.Tensor, optional): An optional tensor of labels, not used in the sampling process. Defaults to None.
            sample_frac (float, optional): The fraction of the original data size to include in each variate. Defaults to 0.3.
            sample_number (int, optional): The number of variates (samples) to generate. Defaults to 100.
    
        Returns:
            list: A list of tensors, where each tensor is a variate (sample) drawn from `X`.
    """
    assert sample_frac < 1
    total_size = X.size(0)
    n_samples = int(total_size * sample_frac)
    return [sample_from_Xy_tensors(X, n_samples) for _ in range(sample_number)]

def create_joint(plot=False, sample_size=1000,
                 arc_size=1000):
    """
    Generates a joint dataset consisting of two Gaussian blobs and a noisy arc.
        
        This function creates a synthetic dataset composed of two distinct Gaussian
        clusters and a semi-circular arc with added noise. This type of dataset
        is useful for evaluating the ability of the GAN to capture different data
        distributions and complex shapes, which is important for generating
        high-quality synthetic data that preserves the characteristics of real-world
        datasets. The optional plotting functionality allows for visual inspection
        of the generated data and its kernel density estimation (KDE).
        
        Args:
          plot (bool): If True, plots the generated data and its KDE. Defaults to False.
          sample_size (int): The number of samples to generate for each Gaussian blob. Defaults to 1000.
          arc_size (int): The number of points to generate for the arc. Defaults to 1000.
        
        Returns:
          torch.Tensor: A tensor containing all generated samples (Gaussian blobs and arc).
    """
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

    if plot:
        # Perform KDE
        kde = gaussian_kde(all_samples.T)
        x_grid, y_grid = np.mgrid[-10:10:100j, -10:10:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        density = kde(positions).reshape(x_grid.shape)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(x_grid, y_grid, density, levels=50, cmap="viridis")
        plt.scatter(all_samples[:, 0], all_samples[:, 1], s=5, color="white", alpha=0.5)
        plt.title("KDE of 2 Gaussians + Noisy Arc")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return torch.tensor(all_samples, dtype=torch.float)

# Generate real data (Gaussian mixture)
def get_real_data(batch_size):
    """
    Generates a batch of real data points from a mixture of two Gaussians.
        
        The data is sampled from two Gaussian distributions: one centered at [3, 3]
        and the other at [-5, -5], both with a standard deviation of 1.0. The
        method randomly chooses between these two distributions for each data point
        in the batch. This serves as the real data distribution that the generator
        will attempt to mimic.
        
        Args:
            batch_size (int): The number of data points to generate in the batch.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, 2) containing the generated
                data points. The data type is torch.float.
    """
    mix = np.random.choice(2, size=(batch_size,))
    data = np.zeros((batch_size, 2))
    data[mix == 0] = np.random.normal(loc=[3, 3], scale=1.0, size=(np.sum(mix == 0), 2))
    data[mix == 1] = np.random.normal(loc=[-5, -5], scale=1.0, size=(np.sum(mix == 1), 2))
    return torch.tensor(data, dtype=torch.float)

def remove_anomalies_iqr(x: pd.DataFrame, y: pd.DataFrame):
    """
    Removes outliers from the input DataFrame using the IQR method to improve the quality and stability of synthetic data generation.
    
        This function calculates the first quartile (Q1), third quartile (Q3), and
        interquartile range (IQR) of the input DataFrame `x`. It then identifies and
        removes rows containing values outside the range of Q1 - 1.5 * IQR to
        Q3 + 1.5 * IQR. This step is crucial for ensuring that the generated synthetic
        data is not skewed by extreme values present in the original dataset, leading
        to more representative and reliable synthetic samples.
    
        Args:
            x: The DataFrame to remove anomalies from.
            y: The target DataFrame.
    
        Returns:
            tuple: A tuple containing two DataFrames:
                - The first DataFrame contains the data from `x` with anomalies removed.
                - The second DataFrame contains the corresponding target values from `y`.
    """
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1

    real_data_scaled_no_anomalies_indexes = x[~((x < (Q1 - 1.5 * IQR)) |(x > (Q3 + 1.5 * IQR))).any(axis=1)].index
    real_data_scaled_no_anomalies = x.iloc[real_data_scaled_no_anomalies_indexes, :].values
    real_data_scaled_no_anomalies = real_data_scaled_no_anomalies.values
    y = pd.DataFrame(y[real_data_scaled_no_anomalies.index],
                     columns=["target"])

    # real_data_scaled_no_anomalies["target"] = y[real_data_scaled_no_anomalies.index]
    # real_data_scaled_no_anomalies.to_csv("california_scaled_no_anomalies.csv", index=False)
    return real_data_scaled_no_anomalies, y