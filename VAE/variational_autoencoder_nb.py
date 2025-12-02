#!/usr/bin/env python3

"""
Variational Autoencoder for Isoform Expression Prediction
"""
# Commented out IPython magic to ensure Python compatibility.
from typing import *
from plotting import make_vae_plots
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
import math
import torch
import os 
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
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

# ------------------------------
# Helper function for unique file paths
# ------------------------------

def get_unique_path(base_path):
    """
    If base_path exists, append _2, _3, _4, ... before the extension.
    Example:
        vae_training_epoch010.png
        vae_training_epoch010_2.png
        vae_training_epoch010_3.png
    """
    if not os.path.exists(base_path):
        return base_path
    
    root, ext = os.path.splitext(base_path)
    counter = 2
    new_path = f"{root}_{counter}{ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{root}_{counter}{ext}"
    
    return new_path

FIG_DIR = "vae_figs"
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------
# Implementation of the Gaussian distribution with reparameterization trick
# -----------------------------

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
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
# Implementation of the Negative Binomial distribution
# ------------------------------

class NegativeBinomial(Distribution):
    """
    Negative Binomial with mean `mu` and inverse-dispersion `theta` (>0).

    We implement:
      - log_prob(x)  : exact NB log-likelihood (used for training)
      - sample()     : robust NB(mu) approximation (used only for generation)
    """
    arg_constraints = {
        "mu": constraints.positive,
        "theta": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu: Tensor, theta: Tensor, eps: float = 1e-8, validate_args=None):
        # Broadcast mu and theta to the same shape
        self.mu, self.theta = broadcast_all(mu, theta)
        self.eps = eps

        batch_shape = self.mu.size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=validate_args)

    def _nb_base_dist(self):
        """
        Build the underlying torch.distributions.NegativeBinomial
        with (total_count, probs) parameterization.

        For consistency with:
          E[X]   = mu
          Var[X] = mu + mu^2 / theta
        we use:
          total_count = theta
          probs       = theta / (theta + mu)
        """
        probs = self.theta / (self.theta + self.mu + self.eps)
        return torch.distributions.NegativeBinomial(
            total_count=self.theta,
            probs=probs
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        base_dist = self._nb_base_dist()
        return base_dist.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        base_dist = self._nb_base_dist()
        return base_dist.sample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mu + (self.mu ** 2) / (self.theta + self.eps)


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

X = data["X_gene"].float().cpu()
#Y = torch.log1p(data["Y_tx"].float()).cpu()

N, G = X.shape
#_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}")


# ------------------------------
# Create train/val/test splits
# ------------------------------

all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel, random_state=SEED, shuffle=True)


# ------------------------------
# Create datasets
# ------------------------------
full_dataset = TensorDataset(X)

dset_train = Subset(full_dataset, tr_idx)
dset_test  = Subset(full_dataset, te_idx)
dset_val   = Subset(full_dataset, va_idx)

# ------------------------------
# DataLoaders
# ------------------------------

batch_size = 64
eval_batch_size = 128

# A stratified sampler is a sampling method that builds batches 
# while preserving the class proportions of the original dataset.
# Unsupervised learning: no class labels.

train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dset_test, batch_size=eval_batch_size, shuffle=False)
val_loader = DataLoader(dset_val, batch_size=eval_batch_size, shuffle=False)

# We will return the number of batches in each loader for verification (total samples / batch size)
print(f"[INFO] Loader sizes: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")


print("[INFO] Extracting VAE latents for all samples...")
full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

# ------------------------------
# Building the model
# ------------------------------

"""
When defining the model the latent layer must act as a bottleneck of information, so that we 
ensure that we find a strong internal representation. We initialize the VAE with 1 hidden layer 
in the encoder and decoder using relu units as non-linearity.
"""

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Gaussian likelihood observation model `p_\theta(x | z) = NB(x | \mu_\theta(z), \theta_\theta(z) I)
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape: torch.Size, latent_features: int,
                 input_dropout_p: float = 0.1, hidden_dropout_p: float = 0.0):
        super().__init__()


        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        # Dropout layer on the input
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=H1),
            nn.ReLU(),
            nn.Linear(in_features=H1, out_features=H2),
            nn.ReLU(),
            nn.Linear(in_features=H2, out_features=H3),
            nn.ReLU(),
            nn.Linear(in_features=H3, out_features=2*latent_features)
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i NB(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=H3),
            nn.ReLU(),
            nn.Linear(in_features=H3, out_features=H2),
            nn.ReLU(),
            nn.Linear(in_features=H2, out_features=H1),
            nn.ReLU(),
            nn.Linear(in_features=H1, out_features=2*self.observation_features)
        )

        # *** Initialize the final layer for stable mu/theta ***
        # The last layer is responsible for raw_mu and raw_theta.
        # Initialize bias to a small positive value (e.g., log(1e-2)) so mu starts > 0.
        # Initialize weights to be small.
        final_layer = self.decoder[-1]
        
        # Small weight initialization
        nn.init.normal_(final_layer.weight, mean=0., std=1e-4)
        
        # Bias initialization: set raw_mu bias to ensure exp(bias) starts at a reasonable value (e.g., 1e-2)
        # The layer output is [raw_mu, raw_theta]. raw_mu occupies the first half of the bias vector.
        half_out_features = final_layer.out_features // 2
        nn.init.constant_(final_layer.bias[:half_out_features], -5.0) # bias for raw_mu: exp(-5.0) ~ 0.0067
        nn.init.constant_(final_layer.bias[half_out_features:], 0.0)  # bias for raw_theta
        # ----------------------------------------------------------------------

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

    def observation_model(self, z: Tensor) -> Distribution:
        """
        return the distribution p(x|z) as Negative Binomial with mean mu(z)
        and inverse-dispersion theta(z) > 0.
        """
        obs_params = self.decoder(z)
        # split into two parts: one for mu, one for theta
        raw_mu, raw_theta = obs_params.chunk(2, dim=-1)

        # mu is the mean of the Negative Binomial (must be > 0).
        # torch.exp ensures strict positivity, preventing the "lambda >= 0" error.
        mu = softplus(raw_mu) + 1e-4
        
        # theta is the inverse-dispersion parameter (must be > 0).
        # softplus is fine here, but we keep the epsilon for safety.
        theta = softplus(raw_theta) + 1e-4

        # reshape back to input shape
        mu = mu.view(-1, *self.input_shape)
        theta = theta.view(-1, *self.input_shape)

        return NegativeBinomial(mu, theta)


    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x_counts = x.view(x.size(0), -1)

        # Transformamos para el encoder: log1p
        x_enc = torch.log1p(x_counts)

        # apply input dropout
        x_enc = self.input_dropout(x_enc)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x_enc)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = NB(x | g(z))
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

# ---- initialize the VAE -----
# Define internal hidden dimensions
H1 = 640
H2 = 320  
H3 = 160
latent_features = 80
vae = VariationalAutoencoder(torch.Size([G]), latent_features)
print(f"[INFO] latent features: {latent_features}, hidden layers: {H1}, {H2}, {H3}")

"""
Implementation of the ELBO and beta ELBO
"""

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: get the mean over all dimensions. 
    Change from previously where we got the sum over all dimensions."""
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

# Sanity check before training
print("[INFO] Sanity check of the VAE and Variational Inference...")
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
epoch = 0
num_epochs = 100

beta =  1.0
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=3e-5)
max_grad_norm = 1.0

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

print(f"[INFO]: beta={beta}, learning rate={optimizer.param_groups[0]['lr']}, num_epochs={num_epochs}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

# Store generated samples
generated_samples_all_epochs = []  

# --- Training phase: update of the model parameters to minimize the loss function ---
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

    # --- Validation: see how well the model generalizes on unseen data ---
    vae.eval()
    with torch.no_grad(): # do not update the weights
        val_epoch_data = defaultdict(list)
        for (x_val,) in val_loader:
            x_val = x_val.to(device)
            loss, diagnostics, _ = vi(vae, x_val)
            for k, v in diagnostics.items():
                val_epoch_data[k].append(v.mean().item())
        for k, v in val_epoch_data.items():
            validation_data[k].append(np.mean(v))

    # --- Generation: demonstrate the VAE's ability to generate new samples ---
    with torch.no_grad():
        gen_outputs = vae.sample_from_prior(batch_size=64)
        x_generated = gen_outputs['px'].sample().cpu()
        generated_samples_all_epochs.append(x_generated)

    print(f"Epoch {epoch:03d} | loss={loss.item():.3f} | elbo={diagnostics['elbo'].mean().item():.3f} | kl={diagnostics['kl'].mean().item():.3f}")


    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    if epoch == num_epochs:
        fig_path = os.path.join(FIG_DIR, f"vae_training_epoch{epoch:03d}.png")
        fig_path = get_unique_path(fig_path)  
        make_vae_plots(vae, x, outputs, training_data, validation_data,
                    save_path=fig_path)
        print(f"[INFO] Saved VAE training plot to {fig_path}")


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

""" 
# Ensure the model is in evaluation mode
vae.eval()

# Generate latents for the three splits
print("[INFO] Generating latent representations for all data splits...")

# 1. Training Latents
Z_train = get_vae_latents(vae, train_loader, device, use_mean=True)
print(f"[INFO] Z_train shape: {Z_train.shape}")

# 2. Validation Latents
Z_val = get_vae_latents(vae, val_loader, device, use_mean=True)
print(f"[INFO] Z_val shape: {Z_val.shape}")

# 3. Test Latents (Reserved for final, unbiased evaluation)
Z_test = get_vae_latents(vae, test_loader, device, use_mean=True)
print(f"[INFO] Z_test shape: {Z_test.shape}") """

Z_all   = get_vae_latents(vae, full_loader, device, use_mean=True)   # shape (N, latent_dim)
print(f"[INFO] Z_all shape: {Z_all.shape}") 

bh   = os.environ.get("BLACKHOLE")
user = os.environ.get("USER")

if bh is None or user is None:
    raise RuntimeError("Environment variables BLACKHOLE and USER must be set!")

OUTPUT_PT = os.path.join(bh, user, f"vae_latents_all_ld{latent_features}.pt")

""" torch.save(
    {"Z_train": Z_train, "Z_test": Z_test},
    OUTPUT_PT
) """

torch.save(
    {"Z": Z_all},
    OUTPUT_PT
)

print(f"[INFO] Saved VAE latent representations to: {OUTPUT_PT}")