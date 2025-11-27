#!/usr/bin/env python3
"""
Variational Autoencoder for Isoform Expression Prediction
"""
# Commented out IPython magic to ensure Python compatibility.
from typing import *
from plotting import make_vae_plots
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
import math
import torch
import os 
import argparse
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
#from torchvision.transforms import ToTensor
from functools import reduce
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, TensorDataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sns.set_style("whitegrid")

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AMP             = (DEVICE == "cuda")

def parse_args():
    parser = argparse.ArgumentParser(
        description="VAE for gene expression → latent embeddings"
    )

    # Core hyperparameters
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Latent dimensionality z (default: 64)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight beta in beta-VAE (default: 1.0)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate for Adam (default: 3e-5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size (default: 128)")
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="Eval batch size (default: 2x batch-size if 0)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")

    # Optional: name suffix for outputs
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag to add to output filename")

    return parser.parse_args()

args = parse_args()
print("[INFO] Parsed args:", args)

# We observed that the first gradient steps can be quite large, destabilizing training.
# To mitigate this, we reduce the effect of large gradients by averaging over dimensions
# instead of summing when reducing log probabilities.
def reduce(x: Tensor) -> Tensor:
    # Old (too big):
    # return x.view(x.size(0), -1).sum(dim=1)

    # New (more stable):
    return x.view(x.size(0), -1).mean(dim=1)


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`eps ~ N(0, I)`"""
        # creates an empty tensor with the same shape as mu and fills it with values from a N(0, 1)
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return -0.5 * (
            ((z - self.mu) ** 2) / (self.sigma ** 2)
            + 2 * self.sigma.log()
            + math.log(2 * math.pi)
        )



# ------------------------------
# Data path (portable)
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")

if DATA_PT is None:
    # Fall back to $BLACKHOLE/$USER/data.pt
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export DATA_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)


X = data["Xg_log1p"].float().cpu()
#Y = torch.log1p(data["Y_tx"].float()).cpu()


N, G = X.shape
#_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}")


# Define the train, test and validation sets
all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel, random_state=SEED, shuffle=True)


# ------------------------------
# Create datasets
# ------------------------------
full_dataset = TensorDataset(X)

dset_train = Subset(full_dataset, trval_idx)
dset_test  = Subset(full_dataset, te_idx)
#dset_val   = Subset(full_dataset, va_idx)

# ------------------------------
# DataLoaders
# ------------------------------

batch_size = args.batch_size
eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else 2 * batch_size

print(f"[INFO] batch_size={batch_size}, eval_batch_size={eval_batch_size}")

train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dset_test,  batch_size=eval_batch_size, shuffle=False)

print(f"[INFO] Loader sizes: train={len(train_loader)}  val={len(test_loader)}")

print("[INFO] Extracting VAE latents for all samples...")
full_loader = DataLoader(full_dataset, batch_size=eval_batch_size, shuffle=False)


"""
Building the model
When defining the model the latent layer must act as a bottleneck of information, so that we ensure that we find a strong internal representation. We initialize the VAE with 1 hidden layer in the encoder and decoder using relu units as non-linearity.
"""

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Gaussian likelihood observation model `p_\theta(x | z) = N(x | \mu_\theta(z), \sigma^2_\theta(z) I)`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)


        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=128, out_features=2*latent_features) # <- note the 2*latent_features
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2*self.observation_features)
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)

        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        obs_params = self.decoder(z)
        mu, log_sigma = obs_params.chunk(2, dim=-1)
        # reshape the output to the input shape
        mu = mu.view(-1, *self.input_shape) 
        log_sigma = log_sigma.view(-1, *self.input_shape)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = N(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}


    def sample_from_prior(self, batch_size:int=128):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = N(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'z': z}

# initialize the VAE
latent_features = args.latent_dim
print(f"[INFO] Using latent_dim={latent_features}")
vae = VariationalAutoencoder(torch.Size([G]), latent_features)


"""
Implementation of the ELBO and beta ELBO
"""

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo =  log_px - kl 
        beta_elbo = log_px - self.beta * kl 

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

vi = VariationalInference(beta=1.0)
loss, diagnostics, outputs = vi(vae, X)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")

"""## Training and Evaluation

### Initialize the model, evaluator and optimizer
"""

from collections import defaultdict
# define the models, evaluator and optimizer

# Evaluator: Variational Inference
beta = args.beta
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
max_grad_norm = args.max_grad_norm

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0
num_epochs = args.epochs

print(f"[INFO] beta={beta}, lr={args.lr}, epochs={num_epochs}, max_grad_norm={max_grad_norm}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

# Store generated samples
generated_samples_all_epochs = []  

# training..
while epoch < num_epochs:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known
    for (x,) in train_loader:
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
        optimizer.step()

        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]


    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # --- Validation ---
    vae.eval()
    with torch.no_grad():
        x_val, = next(iter(test_loader))
        x_val = x_val.to(device)
        loss, diagnostics, outputs = vi(vae, x_val)
        for k, v in diagnostics.items():
            validation_data[k].append(v.mean().item())

    # --- Generation ---
    with torch.no_grad():
        gen_outputs = vae.sample_from_prior(batch_size=64)
        x_generated = gen_outputs['px'].sample().cpu()
        generated_samples_all_epochs.append(x_generated)

    print(f"Epoch {epoch:03d} | loss={loss.item():.3f} | elbo={diagnostics['elbo'].mean().item():.3f} | kl={diagnostics['kl'].mean().item():.3f}")


    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    make_vae_plots(vae, x, outputs, training_data, validation_data)

def get_vae_latents(vae: VariationalAutoencoder,
                    loader: DataLoader,
                    device: torch.device,
                    use_mean: bool = True) -> torch.Tensor:
    """
    Compute latent representations for all samples in `loader`.

    If use_mean=True, returns mu(x) (deterministic embedding).
    If use_mean=False, returns a sample z ~ q(z|x).
    """
    vae.eval()
    all_z = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            x = x.view(x.size(0), -1)

            qz = vae.posterior(x)   # q(z|x) = N(mu, sigma)

            if use_mean:
                z_batch = qz.mu     # posterior mean as embedding
            else:
                z_batch = qz.rsample()  # sample using reparam trick

            all_z.append(z_batch.cpu())

    return torch.cat(all_z, dim=0)

""" Z_train = get_vae_latents(vae, train_loader, device, use_mean=True)
Z_test  = get_vae_latents(vae, test_loader,  device, use_mean=True)
print(f"[INFO] Z_train shape: {Z_train.shape}, Z_test shape: {Z_test.shape}")  # should be (N_train, latent_features), (N_test, latent_features) """

Z_all   = get_vae_latents(vae, full_loader, device, use_mean=True)   # shape (N, latent_dim)
print(f"[INFO] Z_all shape: {Z_all.shape}") 

if bh is None or user is None:
    raise RuntimeError("Environment variables BLACKHOLE and USER must be set!")

# build a nice suffix for the file name
lr_str = f"{args.lr:.0e}".replace("-", "m")  # e.g. 3e-04 -> 3e-04m if you want, or just use raw str(args.lr)
suffix_parts = [
    f"z{args.latent_dim}",
    f"b{args.beta}",
    f"lr{lr_str}",
    f"bs{args.batch_size}",
]
if args.tag:
    suffix_parts.append(args.tag)

suffix = "_".join(suffix_parts)
filename = f"vae_latents_all_{suffix}.pt"
OUTPUT_PT = os.path.join(bh, user, filename)

torch.save({"Z": Z_all}, OUTPUT_PT)
print(f"[INFO] Saved VAE latent representations to: {OUTPUT_PT}")