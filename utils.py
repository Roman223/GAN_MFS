from scipy.spatial.distance import jensenshannon, cosine

from ripser import ripser
from persim import wasserstein
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import tqdm
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

def sample_from_Xy_tensors(X:torch.Tensor, n_samples):
    indices = torch.randperm(X.size(0))
    indices_trunc = indices[:n_samples]
    X_sampled_tensor = X[indices_trunc[:n_samples]]
    # y_sampled_tensor = y[indices_trunc[:n_samples]]
    # return X_sampled_tensor, y_sampled_tensor
    return X_sampled_tensor

def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]

def convert_result(results):
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
    """Replace inf in the diagram with a finite value."""
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
    dgms_X = ripser(X, maxdim=maxdim)['dgms']
    dgms_Y = ripser(Y, maxdim=maxdim)['dgms']
    h0 = wasserstein(dgms_X[0], dgms_Y[0])
    h1 = wasserstein(dgms_X[1], dgms_Y[1]) if len(dgms_X) > 1 else None
    return h0, h1

def correlation_matrix_distance(df1: pd.DataFrame,
                                df2: pd.DataFrame,
                                metric: str = 'frobenius') -> float:
    """
    Compute distance between the correlation matrices of two dataframes.

    Parameters:
        df1, df2: pandas.DataFrame
            DataFrames with same columns.
        metric: str
            Distance metric: 'frobenius', 'euclidean', 'cosine', or 'spectral'.

    Returns:
        float: distance between correlation matrices.
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
    return {"cosine_dist_corr_matrix": correlation_matrix_distance(synth, test, metric="cosine")}

def compute_regression_performance(X,
                                   y,
                                   X_test,
                                   y_test):
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
    assert sample_frac < 1
    total_size = X.size(0)
    n_samples = int(total_size * sample_frac)
    return [sample_from_Xy_tensors(X, n_samples) for _ in range(sample_number)]

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
    mix = np.random.choice(2, size=(batch_size,))
    data = np.zeros((batch_size, 2))
    data[mix == 0] = np.random.normal(loc=[3, 3], scale=1.0, size=(np.sum(mix == 0), 2))
    data[mix == 1] = np.random.normal(loc=[-5, -5], scale=1.0, size=(np.sum(mix == 1), 2))
    return torch.tensor(data, dtype=torch.float)

def remove_anomalies_iqr(x: pd.DataFrame, y: pd.DataFrame):
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