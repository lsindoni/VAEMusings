import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


def multi_gauss_diag_kl_divergence(μ0, σ0, μ1, σ1):
    """
    KL divergence between two multivariate Gaussian distributions with diagonal
    covariances.
    args:
    - μ0: mean of the first Gaussian
    - σ0: standard deviations of the first Gaussian
    - μ1: mean of the second Gaussian
    - σ1: standard deviations of the second Gaussian
    """
    μs = (μ0 - μ1)**2 / σ1 ** 2
    σs = σ0 ** 2 / σ1 ** 2 - 2 * torch.log(σ0 / σ1)
    kl = 0.5 * (μs + σs - 1)
    return kl.sum(axis=-1).mean()


def get_datasets(train=True, test=True, data_path=None, download=False):
    """
    Download (if necessary) and prepare FashionMNIST dataset.
    - arguments:
        train: boolean. If True, get the training set
        test: boolean. It True, get the test set
        data_path: if None, look for the location "./data/" else provide a custom
            location
        download: boolean, if True, downloads the data. Defaults to False
    - outputs:
        a dictionary with two keys, train and test, each of which is a torch Dataset.
    """
    if not data_path:
        data_path = "./data/"
        if not os.path.exists(data_path):
            os.mkdir(data_path)
    out = {}
    if train:
        out["train"] = FashionMNIST(
            root=data_path,
            train=True,
            download=download,
            transform=ToTensor(),
        )
    if test:
        out["test"] = FashionMNIST(
            root=data_path,
            train=False,
            download=download,
            transform=ToTensor(),
        )
    return out


def make_tensor_dataset(nparray):
    """
    transform a ndarray into a TensorDataset
    """
    tensor = torch.tensor(nparray, dtype=torch.float32)
    return TensorDataset(tensor)


class DummyDataset(Dataset):
    """
    A simple dummy dataset containing random numbers.
    args:
    - n_samples
    - *dims
    - offset=0.0
    - scales=1.0

    Data are stored in .data
    """
    def __init__(self, n_samples, *dims, offset=0.0, scale=1.0):
        self.dims = dims
        self.n_samples = n_samples
        self.data = [torch.randn(n_samples, dim) * scale + offset for dim in dims]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        items = [xs[idx] for xs in self.data]
        return tuple(items)


class MCDataset(Dataset):
    """
    Creates a dataset for MC simulations.
    args:
    - n_samples: n_samples in the dataset
    - n_mc_samples: number of mc samples per entry
    - dim: the dimensionality of the samples
    """
    def __init__(self, n_samples, n_mc_samples, dim):
        self.dims = dim
        self.n_samples = n_samples
        self.n_mc_samples = n_mc_samples
        self.xs = torch.randn(n_samples, n_mc_samples, dim)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.xs[idx],


class FFWDNN(nn.Module):
    def __init__(self, *, input_dims, layer_units, activations, name=None):
        assert len(layer_units) == len(activations), "Mismatch in number of layers and number of units"
        super().__init__()
        self.name = name if name else "FFWD_NN"
        self.input_dims = input_dims
        self.output_dims = layer_units[-1]
        self._layers = layer_units
        self._activations = activations
        self._make_submodule_name = lambda x: f"{self.name}_submodule_{i}"
        self.layers = {}
        self.layer_names = []
        current_inputs = input_dims
        for i, units in enumerate(layer_units):
            submodule = nn.Linear(current_inputs, units)
            sub_name = self._make_submodule_name(i)
            self.add_module(sub_name, submodule)
            self.layers[sub_name] = submodule
            self.layer_names.append(sub_name)
            current_inputs = units

    def forward(self, inputs):
        x = inputs
        for sub_name, activation in zip(self.layer_names, self._activations):
            x = self.__getattr__(sub_name)(x)
            x = activation(x)
        return x


class ColliderNN(FFWDNN):
    def __init__(self, *, left_input_dims, right_input_dims, layer_units, activations, name=None):
        name = name if name else "Collider_NN"
        assert len(layer_units) == len(activations), "Mismatch in number of layers and number of units"
        super().__init__(input_dims=(left_input_dims + right_input_dims),
            layer_units=layer_units,
            activations=activations,
            name=name,
        )
        self.left_input_dims = left_input_dims
        self.right_input_dims = right_input_dims
        self.input_dims = [left_input_dims, right_input_dims]

    def forward(self, inputs):
        left_inputs, right_inputs = inputs
        x = torch.cat([left_inputs, right_inputs], axis=1)
        for sub_name, activation in zip(self.layer_names, self._activations):
            x = self.__getattr__(sub_name)(x)
            x = activation(x)
        return x


ACTIVATIONS_LOOKUP = {
    "relu": nn.functional.relu,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
    "none": lambda x: x,
}


CONSTRUCTOR_LOOKUP = {
    "ffwdnn": FFWDNN,
    "collidernn": ColliderNN,
}


# def make_ffwdnn(architecture):
#     activations = [ACTIVATIONS_LOOKUP[activ] for activ in architecture["activations"]]
#     architecture.update({"activations": activations})
#     return FFWDNN(**architecture)


def make_nn(architecture):
    constructor_ = architecture.pop("constructor", "ffwdnn")
    constructor = CONSTRUCTOR_LOOKUP[constructor_]
    activations = [ACTIVATIONS_LOOKUP[activ] for activ in architecture["activations"]]
    architecture.update({"activations": activations})
    return constructor(**architecture)


class FullFFWDModel:
    def __init__(self, module, loss, optimizer, **optimizer_kwargs):
        self.module = module
        self.loss = loss
        self.optimizer = optimizer(module.parameters(), **optimizer_kwargs)
        self.losses = []

    def fit(self, dataset, epochs=1, batch_size=1):
        train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            print(f"epoch: {epoch+1}")
            for minibatch in train_dl:
                xs, ys = minibatch
                self.optimizer.zero_grad()
                pred = self.module(xs)
                loss = self.loss(pred, ys)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.detach().numpy())

    def predict(self, xs):
        return self.module(xs)


class ProbNN(nn.Module):
    def __init__(self, mean_nn, log_sigma_nn):
        super().__init__()
        self.mean = mean_nn
        self.log_sigma = log_sigma_nn
        assert mean_nn.input_dims == log_sigma_nn.input_dims, "mean and covariance have different input shapes"
        assert mean_nn.output_dims == log_sigma_nn.output_dims, "mean and covariance have different output shapes"
        self.input_dims = self.mean.input_dims
        self.output_dims = self.mean.output_dims

    def forward(self, inputs):
        return self.mean(inputs), torch.exp(self.log_sigma(inputs))

    def sample(self, inputs):
        μ, σ = self(inputs)
        ϵs = torch.randn(σ.shape)
        return μ + σ * ϵs

    def log_likelihood(self, visible, latent):
        μ, σ = self(latent)
        exp = - 0.5 * (visible - μ)**2 / σ ** 2
        penalty = - torch.log(σ)
        ll = exp + penalty
        return ll.sum(axis=-1).mean()


def make_prob_nn(architecture):
    mean_nn = make_nn(architecture["μ"])
    log_sigma_nn = make_nn(architecture["log_σ"])
    return ProbNN(mean_nn, log_sigma_nn)


class TProdDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        # super().__init__()
        assert len(dataset1) == len(dataset2), "Datasets have different dimensions"
        self.data1 = dataset1
        self.data2 = dataset2
        self.n_samples = len(dataset1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data1[idx]
        y = self.data2[idx]
        return tuple([x, y])


class VAE(nn.Module):
    def __init__(self, generator, recognition, optimizer=None, n_mc_samples=1, **optimizer_options):
        super().__init__()
        self.generator = generator
        self.recognition = recognition
        self.n_mc_samples = n_mc_samples
        self.visible_dims = self.recognition.input_dims
        self.latent_dims = self.generator.input_dims
        assert self.generator.output_dims == self.visible_dims, "Mismatch with latent dimensions"
        assert self.latent_dims == self.recognition.output_dims
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **optimizer_options)

    def make_vae_dataset(self, dataset):
        n_samples = len(dataset)
        mc_dataset = MCDataset(n_samples, self.n_mc_samples, self.latent_dims)
        self.train_dataset = TProdDataset(dataset, mc_dataset)
        return self.train_dataset

    # def kl_divergence(self, μ_latent, σ_latent):
    #     #l = 0.5 * (μ_latent**2 - 2 * torch.log(σ_latent) + σ_latent ** 2 - 1)
    #     return l.sum(axis=-1).mean()

    def ll_loss(self, μ_latent, σ_latent, visible, latent_noise):
        ll = 0.0
        for idx in range(self.n_mc_samples):
            latent = μ_latent + σ_latent * latent_noise[:, idx, :]
            ll = ll - self.generator.log_likelihood(visible, latent)
        return ll / self.n_mc_samples

    def vae_loss(self, visible, latent_noise):
        μ_l, σ_l = self.recognition(visible)
        μ_p, σ_p = torch.zeros(μ_l.shape), torch.ones(μ_l.shape)
        kl_loss = multi_gauss_diag_kl_divergence(μ_l, σ_l, μ_p, σ_p)
        # kl_loss = self.kl_divergence(μ_l, σ_l)
        self.kl_losses.append(kl_loss.detach().numpy().item())
        ll_loss = self.ll_loss(μ_l, σ_l, visible, latent_noise)
        self.ll_losses.append(ll_loss.detach().numpy().item())
        return ll_loss + kl_loss

    def fit(self, dataset, batch_size=1, epochs=1):
        train_dataset = self.make_vae_dataset(dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
        self.kl_losses = []
        self.losses = []
        self.ll_losses = []
        for epoch in range(epochs):
            for data in train_dataloader:
                (visible, ), (latent_noise,) = data
                self.optimizer.zero_grad()
                loss = self.vae_loss(visible, latent_noise)
                self.losses.append(loss.detach().numpy().item())
                loss.backward()
                self.optimizer.step()

    def plot_losses(self, y_scale=5):
        fig, axs = plt.subplots(1, 3, figsize=(3 * y_scale, y_scale))
        axs[0].plot(self.losses, color="k")
        axs[0].grid()
        axs[0].set_title("Total VAE Loss")
        axs[1].plot(self.ll_losses, color="g")
        axs[1].grid()
        axs[1].set_title("Negative Log-Likelihood")
        axs[2].plot(self.kl_losses, color="gray")
        axs[2].grid()
        axs[2].set_title("KL Loss")
        return fig

    def plot_latent(self, data, y_scale=5, labels=None):
        data = torch.tensor(data, dtype=torch.float32)
        latent = self.recognition.sample(data).detach()
        regenerated = self.generator.sample(latent).detach()
        fig, axs = plt.subplots(1, 2, figsize=(2 * y_scale, y_scale))
        if labels is not None:
            color=labels
        else:
            color="g"
        axs[0].scatter(latent[:, 0], latent[:, 1], c=color)
        axs[0].set_title("Latent")
        axs[1].scatter(regenerated[:, 0], regenerated[:, 1], c=color)
        axs[1].set_title("Reconstructed")

    def sample_generator(self, n_samples=1):
        latent = torch.randn([n_samples, self.latent_dims])
        μ, σ = torch.generator(latents)
        ϵs = torch.randn(μ.shape)
        return μ + σ * ϵs

    def sample_similar(self, examples):
        latent = self.recognition(examples)
        μ, σ = torch.generator(latents)
        ϵs = torch.randn(μ.shape)
        return μ + σ * ϵs



OPTIMIZER_LOOKUP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

# - Conditional VAE


class CVAE(nn.Module):
    def __init__(self, prior, generator, recognition, optimizer, n_mc_samples=1, **optimizer_options):
        super().__init__()
        self.prior = prior
        self.generator = generator
        self.recognition = recognition
        self.n_mc_samples = n_mc_samples
        self.conditional_dims = self.prior.input_dims
        self.visible_dims = self.generator.output_dims
        self.latent_dims = self.prior.output_dims
        self.optimizer = optimizer(self.parameters(), **optimizer_options)

    def make_vae_dataset(self, dataset):
        n_samples = len(dataset)
        mc_dataset = MCDataset(n_samples, self.n_mc_samples, self.latent_dims)
        self.train_dataset = TProdDataset(dataset, mc_dataset)
        return self.train_dataset


    def ll_loss(self, μ_latent, σ_latent, conditions, visible, latent_noise):
        ll = 0.0
        for idx in range(self.n_mc_samples):
            latents = μ_latent + σ_latent * latent_noise[:, idx, :]
            ll = ll - self.generator.log_likelihood(visible, [conditions, latents])
        return ll / self.n_mc_samples

    def vae_loss(self, conditions, visible, latent_noise):
        μ_l, σ_l = self.recognition([conditions, visible])
        μ_p, σ_p = self.prior(conditions)
        kl_loss = multi_gauss_diag_kl_divergence(μ_l, σ_l, μ_p, σ_p)
        self.kl_losses.append(kl_loss.detach().numpy().item())
        ll_loss = self.ll_loss(μ_l, σ_l, conditions, visible, latent_noise)
        self.ll_losses.append(ll_loss.detach().numpy().item())
        return ll_loss + kl_loss

    def fit(self, dataset, batch_size=1, epochs=1):
        train_dataset = self.make_vae_dataset(dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
        self.kl_losses = []
        self.losses = []
        self.ll_losses = []
        for epoch in range(epochs):
            for data in train_dataloader:
                (conditions, visible), (latent_noise,) = data
                conditions, = conditions
                visible, = visible
                self.optimizer.zero_grad()
                loss = self.vae_loss(conditions, visible, latent_noise)
                self.losses.append(loss.detach().numpy().item())
                loss.backward()
                self.optimizer.step()

    def plot_losses(self, y_scale=5):
        fig, axs = plt.subplots(1, 3, figsize=(3 * y_scale, y_scale))
        axs[0].plot(self.losses, color="k")
        axs[0].grid()
        axs[0].set_title("Total VAE Loss")
        axs[1].plot(self.ll_losses, color="g")
        axs[1].grid()
        axs[1].set_title("Negative Log-Likelihood")
        axs[2].plot(self.kl_losses, color="gray")
        axs[2].grid()
        axs[2].set_title("KL Loss")
        return fig

    def plot_latent(self, conditions, visible, y_scale=5, labels=None):
        conds = torch.tensor(conditions, dtype=torch.float32)
        vis = torch.tensor(visible, dtype=torch.float32)
        latent = self.recognition.sample([conds, vis]).detach()
        regenerated = self.generator.sample([conds, latent]).detach()
        fig, axs = plt.subplots(1, 2, figsize=(2 * y_scale, y_scale))
        if labels is not None:
            color=labels
        else:
            color="g"
        axs[0].scatter(latent[:, 0], latent[:, 1], c=color)
        axs[0].set_title("Latent")
        axs[1].scatter(regenerated[:, 0], regenerated[:, 1], c=color)
        axs[1].set_title("Reconstructed")

    def sample_generative(self, conditions):
        μ_l, σ_l = self.prior(conditions)
        ϵs = torch.randn(μ_l.shape)
        latents = μ_l + σ_l * ϵs
        μ, σ = self.generator([conditions, latents])
        ϵs = torch.randn(μ.shape)
        return μ + σ * ϵs

    def sample_similar(self, condition, examples):
        μ  = self.recognition([condition, examples])
        μ, σ = torch.generator(latents)
        ϵs = torch.randn(μ.shape)
        return μ + σ * ϵs



# - Convenience constructors

SCHEMA = ["prior", "generator", "recognition"]

def make_vae(**vae_architecture):
    vae_family = vae_architecture.pop("vae_family", "vae")
    if vae_family.lower() == "vae":
        constructor = VAE
    elif vae_family.lower() == "cvae":
        constructor = CVAE
    else:
        raise ValueError(f"Constructor {vae_family} not known")
    nws = []
    for block in SCHEMA:
        if block in vae_architecture:
            print(f"Adding {block}")
            nw = make_prob_nn(vae_architecture[block])
            nws.append(nw)
    # generator = make_prob_nn(vae_architecture["generator"])
    # recognition = make_prob_nn(vae_architecture["recognition"])
    optimizer = OPTIMIZER_LOOKUP[vae_architecture["optimizer"]]
    return constructor(*nws, optimizer,
               n_mc_samples=vae_architecture["n_mc_samples"],
               **vae_architecture["optimizer_options"],
               )


# ~
