import os.path

import pandas as pd
import torch
from torch.autograd import grad as torch_grad
import tqdm
import matplotlib.pyplot as plt
from aim import Image

import ot

PLOT_GRAPH = False
try:
    from torchviz import make_dot

    PLOT_GRAPH = True
except ImportError:
    import warnings

    warnings.warn("torchviz module required to plot computation graph.")

from torch_topological.nn import WassersteinDistance

try:
    import statsmodels.api as sm
except ImportError:
    import warnings

    warnings.warn("statsmodels module required to plot QQ plots.")

from wgan_gp.pymfe_to_torch import MFEToTorch
from sklearn.decomposition import PCA


class Trainer:
    """
    A base class for training generative adversarial networks (GANs).

    This class provides a basic structure for training GANs, including methods for training
    the discriminator and generator, calculating gradient penalties, and generating samples.
    """

    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        batch_size,
        aim_track,
        gen_model_name,
        disable_tqdm=False,
        gp_weight=10,
        critic_iterations=5,
        device=torch.device("cpu"),
    ):
        """
        Initializes the WGAN-GP trainer, setting up the necessary components for adversarial training
        to generate synthetic data.
        This method prepares the generator and discriminator networks, configures their respective
        optimizers, and establishes the training loop parameters. It's crucial for ensuring that
        the GAN training process is correctly initialized, allowing the generator to learn how to
        create realistic synthetic samples while the discriminator learns to distinguish between
        real and generated data.

        Args:
            generator: The generator network.
            discriminator: The discriminator network.
            gen_optimizer: The optimizer for the generator.
            dis_optimizer: The optimizer for the discriminator.
            batch_size: The batch size for training.
            aim_track: A dictionary for tracking training progress with Aim.
            gen_model_name: The name of the generator model.
            disable_tqdm: Whether to disable tqdm progress bar. Defaults to False.
            gp_weight: The weight of the gradient penalty. Defaults to 10.
            critic_iterations: The number of discriminator iterations per generator iteration.
                Defaults to 5.
            device: The device to use for training (e.g., 'cuda' or 'cpu').
                Defaults to torch.device('cpu').
        """
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
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
        """
        Computes the total gradient norm of a model's parameters.

        Calculates the L2 norm of the gradients across all parameters in the model.
        This is useful for monitoring training and detecting potential issues like
        exploding gradients, ensuring stable training during synthetic data generation.
        By monitoring the gradient norm, we can ensure the generator and discriminator
        are learning effectively and prevent instability, which is crucial for producing
        high-quality synthetic data that preserves the utility of the original dataset.

        Args:
            model: The model whose gradients are to be evaluated.

        Returns:
            float: The total gradient norm.
        """
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _critic_train_iteration(self, data):
        """
        Trains the critic (discriminator) for one iteration to distinguish between real and
        generated data.

        The critic's objective is to maximize the difference between its output for real and
        generated samples, along with a gradient penalty to enforce the Lipschitz constraint.
        This step is crucial for improving the generator's ability to create realistic
        synthetic data by providing a strong adversary.

        Args:
            data (torch.Tensor): A batch of real data samples.

        Returns:
            None. The critic's loss is tracked in `self.D_loss`.
        """
        self.D_opt.zero_grad()
        self.D.train()  # just to be explicit

        # Move real data to device
        data = data.to(
            self.device
        )  # assume self.device is torch.device('cuda' or 'cpu')

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
        """
        Performs a single training iteration for the generator network.

        This involves sampling synthetic data, evaluating the generator's performance based
        on the discriminator's output, and updating the generator's weights to improve the
        quality of the generated samples.

        Args:
            data (torch.Tensor): A batch of real data (not used in this function, but present
                for consistency with discriminator training).

        Returns:
            None. The generator loss is stored in `self.G_loss`.

        Why:
            The generator is trained to produce synthetic data that can fool the discriminator.
            This function updates the generator's parameters to minimize the discriminator's
            ability to distinguish between real and generated data, thereby improving the
            realism and utility of the synthetic data.
        """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss = g_loss

    def _gradient_penalty(self, real_data, generated_data):
        """
        Computes the gradient penalty for WGAN-GP.

        This method calculates the gradient penalty, a regularization term used in Wasserstein
        GANs with gradient penalty (WGAN-GP). By penalizing the norm of discriminator gradients
        with respect to its input, we encourage the discriminator to have a smoother landscape.
        This enforces the Lipschitz constraint, which is crucial for the Wasserstein distance
        to be a valid metric and for stable GAN training, ultimately improving the utility of
        generated samples.

        Args:
            real_data (torch.Tensor): Real data samples.
            generated_data (torch.Tensor): Generated data samples.

        Returns:
            torch.Tensor: The gradient penalty.
        """
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
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)

        self.GP_grad_norm = gradients_norm.mean().item()

        # Compute penalty
        gp = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return gp

    def _train_epoch(self, data_loader):
        """
        Trains the GAN for one epoch using the provided data loader, alternating between
        critic and generator training steps.

        The method iterates through the data loader, performing multiple critic updates for
        each generator update, as determined by `self.critic_iterations`. This ensures the
        critic is well-trained to differentiate between real and generated samples, which
        is crucial for the generator to learn to produce realistic synthetic data.

        Args:
            data_loader: The data loader providing real data samples for training.

        Returns:
            None. The method updates the generator and critic networks in place to improve
            the quality and utility of generated samples.
        """
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
        """
        Trains the GAN model to generate synthetic data that mimics the distribution of the real data.

        The training process involves iteratively updating the generator and discriminator
        networks to improve the quality and realism of the generated samples. The progress
        is monitored and visualized through loss tracking and sample plotting.

        Args:
            data_loader: The data loader providing batches of real data for training.
            epochs: The number of training epochs to perform.
            plot_freq: The frequency (in epochs) at which to generate and plot samples to
                visualize training progress.

        Returns:
            None
        """
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

            plt.scatter(samples[:, 0], samples[:, 1], label="Synthetic", alpha=0.3)
            plt.scatter(
                real_data_sample[:, 0],
                real_data_sample[:, 1],
                label="Real data",
                alpha=0.3,
            )

            if pca:
                plt.title(f"Explained var: {sum(pca.explained_variance_ratio_)}")

            plt.legend()
            plt.close(fig)

            aim_fig = Image(fig)
            if epoch % plot_freq == 0:
                self.aim_track.track(aim_fig, epoch=epoch, name="progress")

            if self.aim_track:
                self.aim_track.track(self.G_loss.item(), name="loss G", epoch=epoch)
                self.aim_track.track(self.D_loss.item(), name="loss D", epoch=epoch)
                self.aim_track.track(
                    self.total_grad_norm(self.G), name="total_norm_G", epoch=epoch
                )
                self.aim_track.track(
                    self.total_grad_norm(self.D), name="total_norm_D", epoch=epoch
                )
                self.aim_track.track(
                    self.GP_grad_norm, name="GP_grad_norm", epoch=epoch
                )

    def sample_generator(self, num_samples):
        """
        Generates synthetic data samples using the generator network to augment the original dataset.

        The generated samples aim to resemble the real data distribution, enhancing the dataset's
        utility for downstream tasks.

        Args:
            num_samples: The number of synthetic samples to generate.

        Returns:
            The generated synthetic data samples.
        """
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data


class TrainerModified(Trainer):
    """
    A modified trainer class for training GANs with Meta-Feature Statistics (MFS) preservation.

    This class extends the base trainer to incorporate Meta-Feature Statistics (MFS)
    into the training process, allowing for targeted preservation of statistical properties
    and enhanced synthetic data quality.
    """

    def __init__(self, mfs_lambda, subset_mfs, target_mfs, sample_number, **kwargs):
        """
        Initializes the TrainerModified class with Meta-Feature Statistics preservation.

        This class configures the training process for the GAN, focusing on preserving
        meta-feature statistics to enhance the utility of generated synthetic data. It sets up
        the parameters that guide the MFS preservation process, ensuring the generated data
        retains key statistical characteristics of the real data.

        Args:
            mfs_lambda (float or list): Lambda value(s) for MFS loss weighting, controlling
                the strength of the meta-feature preservation regularization.
            subset_mfs (list): Subset of meta-features to preserve, defining which statistical
                properties to focus on during training.
            target_mfs (dict): Target MFS distributions, specifying the desired meta-feature
                distributions in the generated data. Defaults to {"other_mfs": 0} if not provided.
            sample_number (int): Number of variates to use during MFS calculation, influencing
                the stability and accuracy of meta-feature estimation.
            **kwargs: Additional keyword arguments passed to the parent Trainer class.

        The method initializes the training process by setting up the meta-feature statistics (MFS)
        parameters. This setup is crucial for guiding the GAN to generate synthetic data that not
        only resembles the real data visually but also maintains its statistical utility. The
        target_mfs parameter allows specifying the desired distribution of meta-features in the
        generated data, ensuring that the synthetic data preserves important statistical properties
        for downstream tasks.
        """
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
        """
        Samples a subset of data points from a given tensor.

        This is useful for creating smaller, representative datasets for tasks such as
        evaluating model performance on a subset of the data or for visualization purposes.

        Args:
            tensor (torch.Tensor): The input tensor from which to sample.
            n_samples (int): The number of data points to sample from the tensor.

        Returns:
            torch.Tensor: A new tensor containing the sampled data points.
                The sampled data maintains the original data's structure while reducing its size,
                which is important for efficient analysis and evaluation.
        """
        indices = torch.randperm(tensor.size(0))
        indices_trunc = indices[:n_samples]
        sampled_tensor = tensor[indices_trunc[:n_samples]]
        return sampled_tensor

    def calculate_mfs_torch(
        self, X: torch.Tensor, y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates the meta-feature statistics (MFS) to quantify statistical properties for
        preserving data utility in GAN-generated synthetic data.

        This method leverages the MFS manager to assess various statistical properties of the
        input tensor X, optionally conditioned on a target tensor y. This helps in understanding
        which statistical characteristics are most important for preserving data utility in
        synthetic samples.

        Args:
            X (torch.Tensor): The input tensor representing the synthetic data features.
            y (torch.Tensor, optional): The target tensor representing the corresponding
                target variable. Defaults to None.

        Returns:
            torch.Tensor: The calculated MFS values, moved to the specified device.
                These values represent various statistical properties of the data (correlation,
                covariance, eigenvalues, etc.), which are used to guide the generator's
                learning process and ensure statistical fidelity.
        """
        return self.mfs_manager.get_mfs(X, y, subset=self.subset_mfs).to(self.device)

    @staticmethod
    def total_grad_norm(model):
        """
        Computes the total gradient norm of a model's parameters.

        Calculates the L2 norm of the gradients across all parameters in the model.
        This is useful for monitoring training and detecting potential issues like
        exploding gradients, ensuring stable training during synthetic data generation.
        By monitoring the gradient norm, we can ensure the generator and discriminator
        are learning effectively and preventing mode collapse, which is crucial for
        producing high-quality synthetic data.

        Args:
            model: The model whose gradients are to be analyzed.

        Returns:
            float: The total gradient norm.
        """
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def compute_loss_on_variates_wasserstein(self, fake_distribution):
        """
        Computes the Wasserstein loss to align generated and real data distributions.

        This method calculates the Wasserstein distance between the target meta-feature
        statistics (MFS) and the MFS generated from the fake data distribution. It first
        calculates the MFS for each variate in the fake distribution, reshapes them, and
        then computes the Wasserstein distance using the specified distance function. This
        loss encourages the generator to produce data with similar statistical properties
        to the real data, enhancing the utility of the synthetic data for downstream tasks.

        Args:
            fake_distribution: A list of tensors representing the generated data distribution.
                Each tensor represents a variate.

        Returns:
            torch.Tensor: The Wasserstein distance between the target MFS and the generated MFS.
        """
        fake_mfs = [self.calculate_mfs_torch(X) for X in fake_distribution]
        fake_mfs = self.reshape_mfs_from_variates(fake_mfs)

        # mfs_to_track = fake_mfs.clone()
        # self.mfs_to_track = mfs_to_track.mean(dim=1).cpu().detach().numpy().round(5).tolist()
        return self.wasserstein_dist_func(self.target_mfs["other_mfs"], fake_mfs)

    @staticmethod
    def reshape_mfs_from_variates(mfs_from_variates: list):
        """
        Reshapes a list of meta-feature statistics from variates into a tensor for comparison.

        The input list `mfs_from_variates` contains MFS values, which are stacked
        and then transposed to create the reshaped tensor. This reshaping
        facilitates the calculation of metrics and topological analysis
        needed to evaluate the quality and utility of the generated synthetic data.

        Args:
            mfs_from_variates: A list of meta-feature statistics from variates.

        Returns:
            torch.Tensor: A reshaped tensor where the first dimension corresponds
                to the variates and the second dimension corresponds to the MFS values.
                This format is required for subsequent analysis and comparison
                of real and synthetic data characteristics.
        """
        stacked = torch.stack(mfs_from_variates)
        reshaped = stacked.transpose(0, 1)
        return reshaped

    def wasserstein_distance_2d(self, x1, x2):
        """
        Compute the Wasserstein distance between two 2D point clouds.

        This method calculates the Earth Mover's Distance (EMD), also known as the
        Wasserstein distance, between two sets of 2D points. It assumes that both
        point clouds have equal weights assigned to each point. This distance is used
        to evaluate how well the generated data distribution matches the real data
        distribution, ensuring the synthetic data retains statistical similarity.

        Args:
            x1 (torch.Tensor): The first point cloud, represented as a batch of 2D points.
            x2 (torch.Tensor): The second point cloud, represented as a batch of 2D points.

        Returns:
            float: The Wasserstein distance between the two point clouds.
        """
        batch_size = x1.shape[0]

        ab = torch.ones(batch_size) / batch_size
        ab = ab.to(self.device)

        M = ot.dist(x1, x2)

        return ot.emd2(ab, ab, M)

    def wasserstein_loss_mfs(self, mfs1, mfs2, average=True):
        """
        Calculates the Wasserstein loss between two sets of meta-feature statistics (MFS).

        This method quantifies the statistical similarity between the real and synthetic
        data distributions by computing the Wasserstein distance between corresponding
        feature pairs in the input MFS sets. This loss is used to train the generator
        to produce synthetic data that closely matches the statistical properties of
        the real data.

        Args:
            mfs1 (torch.Tensor): The first set of meta-feature statistics, representing
                the real data distribution.
            mfs2 (torch.Tensor): The second set of meta-feature statistics, representing
                the synthetic data distribution.
            average (bool, optional): A boolean indicating whether to return the average
                loss (True) or a tensor of individual losses (False). Defaults to True.

        Returns:
            torch.Tensor or float: If `average` is True, returns the average
                Wasserstein loss as a float. Otherwise, returns a tensor containing the
                individual Wasserstein distances for each feature.
        """
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
        """
        Generates synthetic data in one training step and optimizes the generator network.

        This method produces data that is indistinguishable from real data based on the
        discriminator's feedback and the matching of meta-feature statistics. The generator
        is optimized to minimize both adversarial loss and MFS preservation loss.

        Args:
            data (torch.Tensor): A batch of real data used to determine batch size.

        Returns:
            None. The generator's parameters are updated to minimize the combined adversarial
                and meta-feature statistics loss. The losses are also recorded.
        """
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
            mfs_dist = self.wasserstein_loss_mfs(
                fake_mfs, self.target_mfs["other_mfs"], average=False
            )

            loss_mfs = mfs_lambda @ mfs_dist
        elif isinstance(self.mfs_lambda, float):
            mfs_dist = self.wasserstein_loss_mfs(
                fake_mfs, self.target_mfs["other_mfs"], average=True
            )
            loss_mfs = self.mfs_lambda * mfs_dist
        else:
            raise TypeError("mfs_lambda must be either a list or a float")

        g_loss = -d_generated.mean() + loss_mfs

        if PLOT_GRAPH:
            if not os.path.isfile("mod_computation_graph_G_loss.png"):
                make_dot(g_loss, show_attrs=True).render(
                    "mod_computation_graph_G_loss", format="png"
                )

        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss = g_loss
        self.mfs_loss = loss_mfs

    @staticmethod
    def plot_grad_flow(named_parameters, title="Gradient flow"):
        """
        Plots the gradient flow through the layers of a neural network to assess training dynamics.

        This method calculates and visualizes the average gradient magnitude
        for each layer of the network, excluding bias parameters. By observing the gradient flow,
        one can identify layers that might be hindering the learning process due to vanishing
        or exploding gradients, ensuring stable and effective training by maintaining data utility.

        Args:
            named_parameters: An iterator of tuples containing layer names and
                parameter tensors.
            title: The title of the plot. Defaults to "Gradient flow".

        Returns:
            matplotlib.figure.Figure: A matplotlib figure containing the gradient flow plot.
        """
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
        plt.xticks(range(len(layers)), layers, rotation="vertical", fontsize=8)
        plt.xlabel("Layer")
        plt.ylabel("Avg Gradient Magnitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_qq_plot(self, mfs_batch):
        """
        Plots a quantile-quantile (QQ) plot to compare MFS distributions.

        This method generates a QQ plot to visually assess how well the generated MFS from
        a batch matches the distribution of the target MFS. It also plots a histogram of the
        target MFS to visualize its distribution. The QQ plot helps determine if the GAN is
        effectively learning to reproduce the statistical properties of the real data.

        Args:
            mfs_batch: A batch of generated MFS to compare against the target distribution.

        Returns:
            matplotlib.figure.Figure: The matplotlib figure containing the QQ plot.
        """
        detached_target = (
            self.target_mfs["other_mfs"].cpu().detach().numpy().reshape(-1, 2)
        )
        mfs_batch_ = mfs_batch.reshape(-1, 2)
        plt.figure()
        plt.hist(detached_target)
        fig = sm.qqplot_2samples(data1=detached_target, data2=mfs_batch_, line="45")
        plt.tight_layout()
        plt.close(fig)
        return fig

    def train(self, data_loader, epochs, plot_freq):
        """
        Trains the GAN model to generate synthetic data that mimics the statistical properties of the real data.

        The training process involves updating the generator and discriminator networks iteratively
        to improve the quality and utility of the generated samples. The method also tracks
        various metrics and visualizations to monitor the training progress and evaluate the
        performance of the GAN.

        Args:
            data_loader: The data loader providing batches of real data for training.
            epochs: The number of training epochs to perform.
            plot_freq: The frequency (in epochs) at which to generate and track plots for
                monitoring training progress.

        Returns:
            None. The method trains the GAN model in place, updating the generator and
                discriminator networks.
        """
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

            self.aim_track.track(self.G_loss.item(), name="loss G", epoch=epoch)
            self.aim_track.track(self.D_loss.item(), name="loss D", epoch=epoch)
            self.aim_track.track(self.mfs_loss, name="loss MFS", epoch=epoch)
            self.aim_track.track(
                self.total_grad_norm(self.G), name="total_norm_G", epoch=epoch
            )
            self.aim_track.track(
                self.total_grad_norm(self.D), name="total_norm_D", epoch=epoch
            )
            self.aim_track.track(self.GP_grad_norm, name="GP_grad_norm", epoch=epoch)

            if epoch % plot_freq == 0:
                fig = plt.figure()
                if samples.shape[1] > 2:
                    pca = True
                    pca = PCA(n_components=2)
                    samples = pca.fit_transform(samples[:, :-1])
                    real_data_sample = pca.fit_transform(real_data_sample[:, :-1])

                plt.scatter(samples[:, 0], samples[:, 1], label="Synthetic", alpha=0.3)
                plt.scatter(
                    real_data_sample[:, 0],
                    real_data_sample[:, 1],
                    label="Real data",
                    alpha=0.3,
                )

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

                fig_G = self.plot_grad_flow(
                    self.G.named_parameters(), title="G gradient flow"
                )
                fig_D = self.plot_grad_flow(
                    self.D.named_parameters(), title="D gradient flow"
                )

                # fig_mfs_distr = self.plot_qq_plot(
                #     mfs_batch=np.asarray([self.calculate_mfs_torch(X).cpu().detach().numpy() for X in samples]))

                aim_fig_G = Image(fig_G)
                aim_fig_D = Image(fig_D)
                # aim_fig_mfs_distr = Image(fig_mfs_distr)

                # self.aim_track.track(aim_fig_mfs_distr, epoch=epoch, name="qq plot")
                self.aim_track.track(aim_fig_G, epoch=epoch, name="G grad flow")
                self.aim_track.track(aim_fig_D, epoch=epoch, name="D grad flow")

        # self.aim_track["mfs_batch"] = self.mfs_to_track
