import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from pprint import pprint
from models import Generator, Discriminator
from training import Trainer, TrainerModified
import aim
from sklearn.datasets import make_moons, fetch_california_housing, make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from naive_try.utils import create_variates
import seaborn as sns
from naive_try.utils import create_joint, calc_metrics, estimate_marginal_js, calc_utility_metrics

print("Torch cuda status: ", torch.cuda.is_available())


class ToyDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.X[idx]
MFS_ENABLED = True

learning_params = dict(
    plot_freq=100,
    # Hyper parameters
    epochs=1000,
    learning_rate_D=0.0001,
    learning_rate_G=0.0001,
    # Dynamics
    betas=(0.5, 0.9),
    support="abalone",
    gp_weight=10,

    generator_dim=(64, 128, 64),
    discriminator_dim=(64, 128, 64) ,
    emb_dim=32,

)

if MFS_ENABLED:
    learning_params |= dict(
        mfs_lambda = .1,
        # subset_mfs = ["cor", "cov", "eigenvalues", "iq_range",
        #                  "mad", "mean", "median", "sd", "var", "sparsity"],
        subset_mfs=["cor", "cov", "mean", "var", "eigenvalues", "iq_range"],
        # subset_mfs=["mean", "var"],
        sample_number=10,
        sample_frac=0.5,
    )
critic_iterations = [1]

# print(f"len mfs {len(learning_params['subset_mfs'])}")
# real_data = create_joint(sample_size=1500, arc_size=2000, plot=False)

real_data_scaled_no_anomalies = pd.read_csv("../data/abalone_cleaned.csv")

print(real_data_scaled_no_anomalies.shape)
real_ds_size = real_data_scaled_no_anomalies.shape
y_no_anomalies = real_data_scaled_no_anomalies.pop("target").values
cols = real_data_scaled_no_anomalies.columns.tolist()
real_data_scaled_no_anomalies = real_data_scaled_no_anomalies.values

# real_data, y = make_regression(n_samples=20000, n_features=4, noise=1, random_state=42)
# real_data, y = make_moons(n_samples=3000, noise=0.05, random_state=42)
# real_data = StandardScaler().fit_transform(real_data)
# cols = ["f1", "f2", "f3", "f4"]

# bunch = fetch_california_housing(as_frame=True)
# real_data, y = bunch.data, bunch.target
# real_data.drop(columns=["AveOccup"], inplace=True)
#
# real_data = real_data.values
#
# cols = ['MedInc', 'HouseAge', 'AveRooms',
#         'AveBedrms', 'Population',
#        'Latitude', 'Longitude']


# print(f"Init data shape: {real_data.shape} + {y.shape}")

# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# scaler = StandardScaler()
# real_data_scaled = scaler.fit_transform(real_data)
# real_data_scaled = pd.DataFrame(real_data_scaled, columns=cols)

print(f"Train shape: {real_ds_size}")
# print(f"Test shape: {test_data.shape}")
# learning_params["batch_size"] = int(real_ds_size[0] * 0.1)
learning_params["batch_size"] = 64
print(learning_params["batch_size"])

eval_number = 4
dead_neurons_storage = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for _ in range(eval_number):
    for critic_iteration in critic_iterations:
        X_train, X_test, y_train, y_test = train_test_split(real_data_scaled_no_anomalies,
                                                            y_no_anomalies,
                                                            test_size=0.3)

        X = np.hstack([X_train, y_train.reshape(-1, 1)])

        print(X.shape, X_train.shape,  X_test.shape, y_train.shape, y_test.shape)
        generator = Generator(embedding_dim=learning_params["emb_dim"], generator_dim=learning_params["generator_dim"],
                              data_dim=X.shape[1]).to(device)

        discriminator = Discriminator(data_dim=X.shape[1],
                                      discriminator_dim=learning_params["discriminator_dim"]).to(device)

        G_optimizer = optim.Adam(generator.parameters(), lr=learning_params["learning_rate_G"],
                                 betas=learning_params["betas"])
        D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_params["learning_rate_D"],
                                 betas=learning_params["betas"])
        tracker = aim.Run(experiment="WGAN-GP")

        tracker['hparams'] = {
            'critic_iterations': critic_iteration,
            **learning_params,
        }

        if MFS_ENABLED:
            trainer = TrainerModified(target_mfs=0, subset_mfs=learning_params["subset_mfs"],
                                      mfs_lambda=learning_params["mfs_lambda"],
                                      generator=generator, discriminator=discriminator,
                                      gen_optimizer=G_optimizer, dis_optimizer=D_optimizer,
                                      device=torch.device("cuda"), critic_iterations=critic_iteration,
                                      batch_size=learning_params["batch_size"], aim_track=tracker,
                                      sample_number=learning_params["sample_number"],
                                      gp_weight=learning_params["gp_weight"],
                                      gen_model_name="wGAN_mfs")
        else:
            trainer = Trainer(generator=generator, discriminator=discriminator,
                              gen_optimizer=G_optimizer, dis_optimizer=D_optimizer,
                              device=torch.device("cuda"), critic_iterations=critic_iteration,
                              batch_size=learning_params["batch_size"], aim_track=tracker,
                              gen_model_name="Vanilla_WGAN")

        X = torch.tensor(X, dtype=torch.float)
        # mfs_distr_real = trainer.calculate_mfs_torch(X)
        # real_eigvals = mfs_distr_real[0].cpu().detach().numpy()
        # real_mean = mfs_distr_real[1].cpu().detach().numpy()
        # real_var = mfs_distr_real[2].cpu().detach().numpy()
        if MFS_ENABLED:
            variates = create_variates(X,
                                       sample_number=learning_params["sample_number"],
                                       sample_frac=learning_params["sample_frac"],)

            mfs_distr = [trainer.calculate_mfs_torch(X_sample) for X_sample in variates] # list of Tensors
            mfs_distr = trainer.reshape_mfs_from_variates(mfs_distr)
            # print(mfs_distr.shape)
            target_features = {
                "persistent_diagram": None,
                "other_mfs": mfs_distr
            }

            trainer.target_mfs = target_features

        # eigvals, mean, var = mfs_distr[0].cpu().numpy(), mfs_distr[1].cpu().numpy(), mfs_distr[2].cpu().numpy()
        # eigvals, means, vars = pd.DataFrame(eigvals[:, :2], columns=["dim1", "dim2"]), pd.DataFrame(mean, columns=["dim1", "dim2"]), pd.DataFrame(var, columns=["dim1", "dim2"])
        # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        # p = sns.jointplot(data=eigvals, x="dim1", y="dim2", kind="kde",
        #                   alpha=0.6)
        # p.fig.suptitle("eigvals")
        # plt.scatter(*real_eigvals, marker="*", color="red", s=100, label="real eigvals")
        # plt.tight_layout()
        # plt.savefig("eigvals.png")
        #
        # p = sns.jointplot(data=means, x="dim1", y="dim2", kind="kde", alpha=0.6)
        # p.fig.suptitle("Means")
        # plt.scatter(*real_mean, marker="*", color="red", s=100, label="real mean")
        # plt.tight_layout()
        #
        # p = sns.jointplot(data=vars, x="dim1", y="dim2", kind="kde", alpha=0.6)
        # p.fig.suptitle("Vars")
        # plt.scatter(*real_var, marker="*", color="red", s=100, label="real var")
        # plt.tight_layout()
        #
        # plt.show()
        # raise Exception

        dataset = ToyDataset(X)
        data_loader = DataLoader(dataset, batch_size=learning_params["batch_size"], shuffle=True, num_workers=5)

        trainer.train(data_loader, learning_params["epochs"], plot_freq=learning_params["plot_freq"])

        synthetic_data = trainer.sample_generator(X.shape[0]).cpu().detach().numpy()
        print(synthetic_data.shape)
        metrics = {"real_vs_test": {}, "synth_vs_test": {}}

        true_metrics, synth_metrics = calc_utility_metrics(
            synth=synthetic_data,
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test)

        metrics["real_vs_test"] |= true_metrics[0]
        metrics["synth_vs_test"] |= synth_metrics[0]

        synthetic_data_df = pd.DataFrame(synthetic_data, columns=cols + ["target"])
        test_data_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1, 1)]), columns=cols + ["target"])

        simple_metrics = calc_metrics(synthetic_data_df, test_data_df)  # cosine distance

        metrics |= simple_metrics

        for k, v in estimate_marginal_js(synthetic_data_df, test_data_df).items():
            metrics["js_div_" + k] = v

        tracker["metrics"] = metrics

        sample = pd.DataFrame(
            trainer.sample_generator(real_ds_size[0]).cpu().detach().numpy(),
            columns=cols + ["target"])

        # r = pd.DataFrame(
        #     X,
        #     columns=cols + ["target"])
        # r["synth"] = 0
        # sample["synth"] = 1
        #
        # f = pd.concat([sample, r])
        # sns.pairplot(f, hue="synth")
        # plt.show()
