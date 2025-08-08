import pandas as pd
from sklearn.datasets import make_moons, make_regression, fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import remove_anomalies_iqr
import seaborn as sns
import matplotlib.pyplot as plt

class RegressionDataset:
    def __init__(self, **kwargs):
        if not kwargs:
            kwargs = dict(n_samples=20000, n_features=4, noise=1)

        self.scaler = StandardScaler()

        real_data, self.y = make_regression(random_state=42, **kwargs)
        real_data_scaled = StandardScaler().fit_transform(real_data)

        cols = [f"f{i}" for i in range(real_data.shape[1])]
        self.data = pd.DataFrame(real_data_scaled, columns=cols)

class MoonsDataset:
    def __init__(self, **kwargs):
        if not kwargs:
            kwargs = dict(n_samples=3000, noise=0.05)

        real_data, self.y = make_moons(random_state=42, **kwargs)
        cols = ["f1", "f2"]
        real_data_scaled = StandardScaler().fit_transform(real_data)
        self.data = pd.DataFrame(real_data_scaled, columns=cols)

class CircleDataset:
    def __init__(self, process=True, **kwargs):
        if not kwargs:
            kwargs = dict(sample_size=1000,
                         arc_size=1000)

        # Generate two Gaussian blobs
        mean1 = [3, 3]
        mean2 = [-5, -5]
        cov = [[0.2, 0], [0, 0.2]]
        samples1 = np.random.multivariate_normal(mean1, cov, size=kwargs["sample_size"])
        samples2 = np.random.multivariate_normal(mean2, cov, size=kwargs["sample_size"])

        # Create a perturbed arc (half-circle)
        theta = np.linspace(0, 2 * np.pi, kwargs["arc_size"])
        arc_x = np.cos(theta)
        arc_y = np.sin(theta)
        arc = np.stack([arc_x, arc_y], axis=1)

        # Translate and add noise to arc
        arc = 5.5 * arc  # scale to match distance
        arc[:, 0] -= 1  # center between the two blobs
        arc[:, 1] -= 1  # center between the two blobs
        arc += np.random.normal(scale=0.2, size=arc.shape)

        # Combine all points
        data = np.vstack([samples1, samples2, arc])
        cols = ["f1", "f2"]
        self.data = pd.DataFrame(data, columns=cols)
        # if plot:
        #     # Perform KDE
        #     kde = gaussian_kde(all_samples.T)
        #     x_grid, y_grid = np.mgrid[-10:10:100j, -10:10:100j]
        #     positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        #     density = kde(positions).reshape(x_grid.shape)
        #
        #     # Plot
        #     plt.figure(figsize=(8, 6))
        #     plt.contourf(x_grid, y_grid, density, levels=50, cmap="viridis")
        #     plt.scatter(all_samples[:, 0], all_samples[:, 1], s=5, color="white", alpha=0.5)
        #     plt.title("KDE of 2 Gaussians + Noisy Arc")
        #     plt.xlabel("x")
        #     plt.ylabel("y")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.show()

class CaliforniaDataset:
    def __init__(self, **kwargs):
        real_data, y = fetch_california_housing(as_frame=True, return_X_y=True)
        real_data.drop(columns=["AveOccup"], inplace=True)
        cols = real_data.columns

        real_data_no_anomalies, self.y = remove_anomalies_iqr(real_data, y)

        # print(f"Init data shape: {real_data_no_anomalies.shape} + {self.y.shape}")

        scaler = StandardScaler()
        real_data_scaled = scaler.fit_transform(real_data_no_anomalies)
        data = pd.DataFrame(real_data_scaled, columns=cols)
        self.data = data.set_index(real_data_no_anomalies.index)

reg = CaliforniaDataset()
print(reg.data)
print(reg.y)
# sns.pairplot(reg.data)
# plt.show()