import os.path

import pandas as pd
import torch
from torch.autograd import grad as torch_grad
import tqdm
import matplotlib.pyplot as plt
from aim import Image

import ot
import numpy as np
from torchviz import make_dot

from torch_topological.nn import WassersteinDistance
import statsmodels.api as sm
from naive_try.wgan_gp.pymfe_to_torch import MFEToTorch
from sklearn.decomposition import PCA


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, batch_size, aim_track,
                 gen_model_name, disable_tqdm=False,
                 gp_weight=10, critic_iterations=5, device=torch.device('cpu')):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations

        self.batch_size = batch_size
        self.num_batches_per_epoch = 0

        self.aim_track = aim_track
        self.G.to(self.device)
        self.D.to(self.device)

        self.disable = disable_tqdm
        self.aim_track["hparams"] |= {"gen_model": gen_model_name}

    @staticmethod
    def total_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _critic_train_iteration(self, data):
        """One training step for the critic (discriminator)"""
        self.D_opt.zero_grad()
        self.D.train()  # just to be explicit

        # Move real data to device
        data = data.to(self.device)  # assume self.device is torch.device('cuda' or 'cpu')

        batch_size = data.size(0)

        # Generate fake data
        with torch.no_grad():  # generator isn't trained here, so we can disable grad
            generated_data = self.sample_generator(batch_size)
            generated_data = generated_data.to(self.device)

        # Detach to be safe (in case generator outputs are connected to autograd graph)
        generated_data = generated_data.detach()

        # Discriminator outputs
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)

        # Compute WGAN-GP loss
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty

        # Backprop
        d_loss.backward()
        self.D_opt.step()

        # Track loss
        self.D_loss = d_loss

    def _generator_train_iteration(self, data):
        """One training step for the generator"""
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss = g_loss

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Sample interpolation factor
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        # Interpolate between real and fake data
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        # Compute critic output on interpolated data
        prob_interpolated = self.D(interpolated)

        # Compute gradients
        grad_outputs = torch.ones_like(prob_interpolated)
        gradients = torch_grad(outputs=prob_interpolated,
                               inputs=interpolated,
                               grad_outputs=grad_outputs,
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)

        self.GP_grad_norm = gradients_norm.mean().item()

        # Compute penalty
        gp = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return gp

    def _train_epoch(self, data_loader):
        data_iter = iter(data_loader)

        for _ in range(self.num_batches_per_epoch):
            # --- Critic updates ---
            for _ in range(self.critic_iterations):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    data = next(data_iter)
                data = data.to(self.device)  # Ensure data is on the correct device
                self._critic_train_iteration(data)
                self.num_steps += 1

            # --- Generator update ---
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)
            data = data.to(self.device)  # Ensure data is on the correct device
            self._generator_train_iteration(data)

    def train(self, data_loader, epochs, plot_freq):
        pca = False
        pbar = tqdm.tqdm(range(epochs), total=epochs, disable=self.disable)
        self.loss_values = pd.DataFrame()
        self.num_batches_per_epoch = len(data_loader)

        for epoch in pbar:
            self._train_epoch(data_loader)
            pbar.set_description(f"Epoch {epoch}")

            fig = plt.figure()
            real_data_sample = next(iter(data_loader))

            samples = self.sample_generator(self.batch_size).cpu().detach().numpy()
            if samples.shape[1] > 2:
                pca = True
                pca = PCA(n_components=2)
                samples = pca.fit_transform(samples[:, :-1])
                real_data_sample = pca.fit_transform(real_data_sample[:, :-1])

            plt.scatter(samples[:, 0], samples[:, 1],
                        label="Synthetic", alpha=0.3)
            plt.scatter(real_data_sample[:, 0], real_data_sample[:, 1],
                        label="Real data", alpha=0.3)

            if pca:
                plt.title(f"Explained var: {sum(pca.explained_variance_ratio_)}")

            plt.legend()
            plt.close(fig)

            aim_fig = Image(fig)
            if epoch % plot_freq == 0:
                self.aim_track.track(aim_fig, epoch=epoch, name="progress")

            if self.aim_track:
                self.aim_track.track(self.G_loss.item(), name='loss G', epoch=epoch)
                self.aim_track.track(self.D_loss.item(), name='loss D', epoch=epoch)
                self.aim_track.track(self.total_grad_norm(self.G), name="total_norm_G", epoch=epoch)
                self.aim_track.track(self.total_grad_norm(self.D), name="total_norm_D", epoch=epoch)
                self.aim_track.track(self.GP_grad_norm, name="GP_grad_norm", epoch=epoch)

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data


class TrainerModified(Trainer):
    def __init__(self, mfs_lambda, subset_mfs, target_mfs, sample_number,
                 **kwargs):
        super(TrainerModified, self).__init__(**kwargs)
        self.mfs_lambda = mfs_lambda
        self.subset_mfs = subset_mfs

        if not target_mfs:
            target_mfs = {"other_mfs": 0}

        self.target_mfs = target_mfs

        if "other_mfs" in target_mfs.keys():
            if isinstance(target_mfs["other_mfs"], torch.Tensor):
                self.target_mfs["other_mfs"] = target_mfs["other_mfs"].to(self.device)

        self.mfs_manager = MFEToTorch()
        self.wasserstein_dist_func = WassersteinDistance(q=2)
        self.sample_number = sample_number

    @staticmethod
    def sample_from_tensor(tensor, n_samples):
        indices = torch.randperm(tensor.size(0))
        indices_trunc = indices[:n_samples]
        sampled_tensor = tensor[indices_trunc[:n_samples]]
        return sampled_tensor

    def calculate_mfs_torch(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.mfs_manager.get_mfs(X, y, subset=self.subset_mfs).to(self.device)

    @staticmethod
    def total_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def compute_loss_on_variates_wasserstein(self, fake_distribution):
        fake_mfs = [self.calculate_mfs_torch(X) for X in fake_distribution]
        fake_mfs = self.reshape_mfs_from_variates(fake_mfs)

        # mfs_to_track = fake_mfs.clone()
        # self.mfs_to_track = mfs_to_track.mean(dim=1).cpu().detach().numpy().round(5).tolist()
        return self.wasserstein_dist_func(self.target_mfs["other_mfs"], fake_mfs)

    @staticmethod
    def reshape_mfs_from_variates(mfs_from_variates: list):
        stacked = torch.stack(mfs_from_variates)
        reshaped = stacked.transpose(0, 1)
        return reshaped

    def wasserstein_distance_2d(self, x1, x2):
        batch_size = x1.shape[0]

        ab = torch.ones(batch_size) / batch_size
        ab = ab.to(self.device)

        M = ot.dist(x1, x2)

        return ot.emd2(ab, ab, M)

    def wasserstein_loss_mfs(self, mfs1, mfs2, average=True):
        # total = 0
        n_features = mfs1.shape[0]

        wsds = []
        for first, second in zip(mfs1, mfs2):
            wsd = self.wasserstein_distance_2d(first, second)
            wsds.append(wsd)
            # total += wsd

        # print_debug = [[i, j.cpu().detach()] for i, j in zip(self.subset_mfs, wsds)]
        # print(*print_debug,
        #       sep='\n')
        if average:
            return sum(wsds) / n_features
        else:
            return torch.stack(wsds).to(self.device)

    def _generator_train_iteration(self, data):
        """One training step for the generator"""
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_variates = []
        for _ in range(self.sample_number):
            generated_data = self.sample_generator(batch_size)

            generated_data.requires_grad_(True)
            generated_data.retain_grad()

            # generated_data = generated_data.to(self.device)
            generated_variates.append(generated_data)

        # Calculate loss and optimize
        d_generated = self.D(generated_variates[0])

        fake_mfs = [self.calculate_mfs_torch(X) for X in generated_variates]
        fake_mfs = self.reshape_mfs_from_variates(fake_mfs)

        if isinstance(self.mfs_lambda, list):
            mfs_lambda = torch.Tensor(self.mfs_lambda).to(self.device)
            mfs_dist = self.wasserstein_loss_mfs(fake_mfs, self.target_mfs["other_mfs"], average=False)

            loss_mfs = mfs_lambda @ mfs_dist
        elif isinstance(self.mfs_lambda, float):
            mfs_dist = self.wasserstein_loss_mfs(fake_mfs, self.target_mfs["other_mfs"], average=True)
            loss_mfs = self.mfs_lambda * mfs_dist
        else:
            raise TypeError("mfs_lambda must be either a list or a float")

        g_loss = - d_generated.mean() + loss_mfs

        if not os.path.isfile("mod_computation_graph_G_loss.png"):
            make_dot(g_loss, show_attrs=True).render("mod_computation_graph_G_loss", format="png")

        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss = g_loss
        self.mfs_loss = loss_mfs

    @staticmethod
    def plot_grad_flow(named_parameters, title="Gradient flow"):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())

        fig = plt.figure(figsize=(10, 5))
        plt.plot(ave_grads, alpha=0.7, marker="o", color="c")
        plt.hlines(0, 0, len(ave_grads), linewidth=1, color="k")
        plt.xticks(rotation="vertical")
        plt.xticks(range(len(layers)), layers, rotation='vertical', fontsize=8)
        plt.xlabel("Layer")
        plt.ylabel("Avg Gradient Magnitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_qq_plot(self, mfs_batch):
        detached_target = self.target_mfs["other_mfs"].cpu().detach().numpy().reshape(-1, 2)
        mfs_batch_ = mfs_batch.reshape(-1, 2)
        plt.figure()
        plt.hist(detached_target)
        fig = sm.qqplot_2samples(data1=detached_target, data2=mfs_batch_, line="45")
        plt.tight_layout()
        plt.close(fig)
        return fig

    def train(self, data_loader, epochs, plot_freq):
        pca = False
        self.mfs_manager.change_device(self.device)
        pbar = tqdm.tqdm(range(epochs), total=epochs, disable=self.disable)
        self.loss_values = pd.DataFrame()
        self.num_batches_per_epoch = len(data_loader)

        for epoch in pbar:
            self._train_epoch(data_loader)
            pbar.set_description(f"Epoch {epoch}")

            real_data_sample = next(iter(data_loader))
            # samples = [self.sample_generator(self.batch_size) for _ in range(self.sample_number)]
            samples = self.sample_generator(self.batch_size).cpu().detach().numpy()
            # for i, mfs_name in enumerate(("cor", "cov", "eigenvalues", "iq_range", "kurtosis", "skewness",
            #                               "mad", "max", "min", "mean", "median", "range", "sd", "var")):
            #     self.aim_track.track(self.diff[i, 0], name=f"{mfs_name}_1", epoch=epoch)
            #     self.aim_track.track(self.diff[i, 1], name=f"{mfs_name}_2", epoch=epoch)

            self.aim_track.track(self.G_loss.item(), name='loss G', epoch=epoch)
            self.aim_track.track(self.D_loss.item(), name='loss D', epoch=epoch)
            self.aim_track.track(self.mfs_loss, name='loss MFS', epoch=epoch)
            self.aim_track.track(self.total_grad_norm(self.G), name="total_norm_G", epoch=epoch)
            self.aim_track.track(self.total_grad_norm(self.D), name="total_norm_D", epoch=epoch)
            self.aim_track.track(self.GP_grad_norm, name="GP_grad_norm", epoch=epoch)

            if epoch % plot_freq == 0:
                fig = plt.figure()
                if samples.shape[1] > 2:
                    pca = True
                    pca = PCA(n_components=2)
                    samples = pca.fit_transform(samples[:, :-1])
                    real_data_sample = pca.fit_transform(real_data_sample[:, :-1])

                plt.scatter(samples[:, 0], samples[:, 1],
                            label="Synthetic", alpha=0.3)
                plt.scatter(real_data_sample[:, 0], real_data_sample[:, 1],
                            label="Real data", alpha=0.3)

                if pca:
                    plt.title(f"Explained var: {sum(pca.explained_variance_ratio_)}")

                plt.legend()
                plt.close(fig)

                aim_fig = Image(fig)
                if epoch % plot_freq == 0:
                    self.aim_track.track(aim_fig, epoch=epoch, name="progress")
                # fig = plt.figure()
                # plt.scatter(samples[0].cpu().detach().numpy()[:, 0], samples[0].cpu().detach().numpy()[:, 1],
                #             label="Synthetic", alpha=0.2)
                # plt.scatter(real_data_sample[:, 0], real_data_sample[:, 1],
                #             label="Real data", alpha=0.2)
                #
                # plt.legend()
                # plt.close(fig)

                # aim_fig = Image(fig)
                # self.aim_track.track(aim_fig, epoch=epoch, name="progress")

                fig_G = self.plot_grad_flow(self.G.named_parameters(), title="G gradient flow")
                fig_D = self.plot_grad_flow(self.D.named_parameters(), title="D gradient flow")

                # fig_mfs_distr = self.plot_qq_plot(
                #     mfs_batch=np.asarray([self.calculate_mfs_torch(X).cpu().detach().numpy() for X in samples]))

                aim_fig_G = Image(fig_G)
                aim_fig_D = Image(fig_D)
                # aim_fig_mfs_distr = Image(fig_mfs_distr)

                # self.aim_track.track(aim_fig_mfs_distr, epoch=epoch, name="qq plot")
                self.aim_track.track(aim_fig_G, epoch=epoch, name="G grad flow")
                self.aim_track.track(aim_fig_D, epoch=epoch, name="D grad flow")

        # self.aim_track["mfs_batch"] = self.mfs_to_track
