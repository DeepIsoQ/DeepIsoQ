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
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
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


from torchvision.transforms import ToTensor
from functools import reduce

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


X = data["Xg_log1p"].float().cpu().numpy()
#Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()


N, G = X.shape
#_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}")


# Define the train, test and validation sets
all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel, random_state=SEED, shuffle=True)
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

batch_size = 64
eval_batch_size = 100
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels))
test_loader  = DataLoader(dset_test, batch_size=eval_batch_size,
                          sampler=stratified_sampler(dset_test.test_labels))


"""
Building the model
When defining the model the latent layer must act as a bottleneck of information, so that we ensure that we find a strong internal representation. We initialize the VAE with 1 hidden layer in the encoder and decoder using relu units as non-linearity.
"""

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
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
            nn.Linear(in_features=256, out_features=self.observation_features)
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
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits, validate_args=False)


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

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}


    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'z': z}


latent_features = 2
vae = VariationalAutoencoder(images[0].shape, latent_features)
print(vae)

"""## Implement a module for Variational Inference

**Exercise 1**: implement `elbo` ($\mathcal{L}$) and `beta_elbo` ($\mathcal{L}^\beta$)
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
        elbo =  log_px - kl # <- your code here
        beta_elbo = log_px - self.beta * kl # <- your code here

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

vi = VariationalInference(beta=1.0)
loss, diagnostics, outputs = vi(vae, images)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")

"""## Training and Evaluation

### Initialize the model, evaluator and optimizer
"""

from collections import defaultdict
# define the models, evaluator and optimizer

# VAE
latent_features = 2
vae = VariationalAutoencoder(images[0].shape, latent_features)

# Evaluator: Variational Inference
beta = 1
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0

"""### Training Loop

**plotting guide**:

* 1st row: Reproducing the figure from the begining of the Notebook.
    * (Left) Data.
    * (Middle) Latent space: the large gray disk reprensents the prior (radius = $2\sigma$), each point represents a latent sample $\mathbf{z}$. The smaller ellipses represent the distributions $q_\phi(\mathbf{z} | \mathbf{x})$  (radius = $2\sigma$). When using $\geq 2$ latent features, dimensionality reduction is applied using t-SNE and only samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ are displayed.
    * (Right) samples from $p_\theta(\mathbf{x} | \mathbf{z})$.

* 2nd row: Training curves

* 2rd row: Latent samples.
    * (Left) Prior samples $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim p(\mathbf{z})$
    * (Middle) Latent Interpolations. For each row: $\mathbf{x} \sim p_\theta(\mathbf{x} | t \cdot \mathbf{z}_1 + (1-t) \cdot \mathbf{z}_2), \mathbf{z}_1, \mathbf{z}_2 \sim p(\mathbf{z}), t=0 \dots 1$.
    * (Right): Sampling $\mathbf{z}$ from a grid [-3:3, -3:3] $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim \operatorname{grid}(-3:3, -3:3)$ (only available for 2d latent space).

**NOTE** this will take a while on CPU.
"""

num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

# training..
while epoch < num_epochs:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in train_loader:
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]


    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()

        # Just load a single batch from the test loader
        x, y = next(iter(test_loader))
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]

    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    make_vae_plots(vae, x, y, outputs, training_data, validation_data)

"""# Analyzing the VAE

### Exercise 1.

1. Implement the class `ReparameterizedDiagonalGaussian` (`log_prob()` and `rsample()`).
2. Import the class `Bernoulli`
3. Implement the class `VariationalInference` (computation of the `elbo` and `beta_elbo`).

### Exercise 2.

**Trainnig and Evaluating a VAE model**

1. Why do we use the reparameterization trick?
2. What available metric can you use to estimate the marginal likelihood ($p_\theta(\mathbf{x})$) ?
3. In the above plots, we display numerous model samples. If you had to pick one plot, which one would you pick to evaluate the quality of a VAE (i.e. using posterior samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ or prior samples $\mathbf{z} \sim p(\mathbf{z})$) ? Why?.
4. How could you exploit the VAE model for classification?

**Answers**:

`[...]`

### Exercise 3.

**Experiment with the VAE model.**

1. Experiment with the number of layers and activation functions in order to improve the reconstructions and latent representation. What solution did you find the best and why?
2. Try to increase the number of digit classes in the training set and analyze the learning curves, latent space and reconstructions. For which classes and why does the VAE fail in reconstructing?  *HINT: Try the combination: `classes=[0, 1, 4, 9]`, to see how well VAE can separate these digits in the latent representation and reconstructions.*
3. Increase the number of units in the latent layer. Does it increase the models representational power and how can you see and explain this? How does this affect the quality of the reconstructions?

**Answers**:

`[...]`

### Exercise 4.

**Analyze the purpose of the KL-term and the $\beta$ parameter.**

1. How does the KL-term, $\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$, work as a regulariser on the distributions over latent variables? *HINT*: When maximising the ELBO, the probability-distance measure is minimised $\operatorname{KL} \rightarrow 0$ so that $q(z|x) \rightarrow p(z) = \mathcal{N}(z|0,I)$ for all examples, x. At $\operatorname{KL} = 0$ variations in x stops having an affect on the latent distribution and latent units are all described by the same distribution, $\mathcal{N}(z|0,I)$, so they produce a noisy output without signal (i.e. $\mathbf{z}=\epsilon \sim \mathcal{N}(0,I)$) to the decoder.
2. Try removing the KL-term (using the $\beta$ parameter) and analyze what happens during training, in the learning curves, latent representation and reconstructions compared to before removing it.
3. What does the loss reduces to? Explain how this can affect a VAE. *HINT*: Compare loss function for AE and VAE, and remember that we can use the pixel-wise binary crossentropy error as the loss in the AEs and for the reconstruction error, $\log p_\theta(\mathbf{x}|\mathbf{z}) = \log \mathcal{B}(\mathbf{x} | g_\theta(z))$, in VAEs.
4. Experiment with different $\beta$ values (e.g. [0.1, 10, 50]) and explain how this affects the reconstruction quality and the latent representations. *HINT* what is the tradeoff between reconstruction error ($\log p_\theta(\mathbf{x} | \mathbf{z})$) and the KL term?

**Answers**:

`[...]`

## Optional exercises

- OPT: Use the original paper http://arxiv.org/pdf/1312.6114v10.pdf or [this blog](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/) to explain what the reparameterization trick does.
- OPT: Look through https://arxiv.org/abs/1611.00712 or https://arxiv.org/abs/1611.01144 and explain how one could instead introduce a categorical latent variable for $z$.
- OPT: Implement the Gumbel softmax trick thereby letting $z$ take a categorical distribution.
- OPT: The VAE is a probablistic model. We could model $p(x,z,y)$ where $y$ is the label information. Explain how this model could handle semi-supervised learning? You can look through the papers https://arxiv.org/pdf/1406.5298.pdf or  https://arxiv.org/pdf/1602.05473v4.pdf or again the two papers on Gumbel softmax.

**Answers**:

`[...]`

# Credits

- Original [Theano/Lasagne tutorial](https://github.com/DeepLearningDTU/nvidia_deep_learning_summercamp_2016/blob/master/lab1/lab1_FFN.ipynb) by
Lars Maaløe ([larsmaaloee](https://github.com/larsmaaloee)),
Søren Kaae Sønderby ([skaae](https://github.com/skaae)), and
Casper Sønderby ([casperkaae](https://github.com/casperkaae)).
- Converted to TensorFlow and updated by Maximillian F. Vording ([maximillian91](https://github.com/maximillian91)).
- Converted to PyTorch and updated by Jesper Wohlert ([wohlert](https://github.com/wohlert)).
- Major update in 2020: focus on the probabilistic interpretation, use of `torch.distributions` and improved visualizations by Valentin Lievin ([vlievin](http://vlievin.github.io)).

# Supplementary Material

## Why a VAE learns a good approximate posterior $q_\phi(\mathbf{z} | \mathbf{x}) \approx p_\theta(\mathbf{z} | \mathbf{x})$

In order to compare the two distributions, let us focus on their KL-divergence (**important**: the divergence is expressed w.r.t $p_\theta(\mathbf{z}|\mathbf{x})$ and not $p(\mathbf{z})$, this is not the KL-divergence studied in the experiments):

$$\mathcal{D}_{\operatorname{KL}}(q_\phi(\mathbf{z} | \mathbf{x}) || p_\theta(\mathbf{z} | \mathbf{x})) = \int_z q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{q_\phi(\mathbf{z} | \mathbf{x}) }{p_\theta(\mathbf{z} | \mathbf{x})}dz = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[\log \frac{q_\phi(\mathbf{z} | \mathbf{x}) }{p_\theta(\mathbf{z} | \mathbf{x})}\right]$$

which is a non-negative distance measure between distributions, so by minimising it wrt. to the parameters in $q_\phi(\mathbf{z} | \mathbf{x}) $, the distribution moves close to our unknown $p_\theta(\mathbf{z} | \mathbf{x})$. But as $p_\theta(\mathbf{z} | \mathbf{x})$ is unknown and would include some rather intractable integrals over neural networks, we can instead get rid of it by expressing it through Bayes rule $p(z|x) = p(x|z)p(z)/p(x)$ and thereby decompose the KL-divergence into our log-likelihood and lower bound:

$$ \mathcal{D}_{\operatorname{KL}}(q_\phi(\mathbf{z} | \mathbf{x})  || p_\theta(\mathbf{z} | \mathbf{x}))
= \int_\mathbf{z} q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{q_\phi(\mathbf{z} | \mathbf{x}) p_\theta(\mathbf{x})}{p_\theta(\mathbf{x} | \mathbf{z})
p(z)}d\mathbf{z}
= \int_\mathbf{z} q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{q_\phi(\mathbf{z} | \mathbf{x}) }{p_\theta(\mathbf{x} | \mathbf{z})p_\theta(\mathbf{x} | \mathbf{z})(\mathbf{z})}d\mathbf{z} + \log p_\theta(\mathbf{x})$$

by seeing that the likelihood, $p_\theta(\mathbf{x})$, is independent of $\mathbf{z}$ and pull it out of the integral. We can flip the sign and fraction in the integral term to recognise it as the negative lower bound

$$\mathcal{D}_{\operatorname{KL}}(q_\phi(\mathbf{z} | \mathbf{x})  || p_\theta(\mathbf{z} | \mathbf{x})) = - \int_z q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{p_\theta(\mathbf{x} | \mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x}) }dz + \log p_\theta(\mathbf{x}) =  -\mathcal{L}(x) + \log p_\theta(\mathbf{x})$$

We then find the log-likelihood to consist of the two terms and hold the inequality

$$\log p_\theta(\mathbf{x}) =  \mathcal{D}_{\operatorname{KL}}(q_\phi(\mathbf{z} | \mathbf{x})  || p_\theta(\mathbf{z} | \mathbf{x})) + \mathcal{L}(\mathbf{x}) \geq \mathcal{L}(\mathbf{x})$$

where the KL-divergence is non-zero and the log-likelihood is $\log p_\theta(\mathbf{x}) \leq 0$. This means that maximising the lower bound from the negative domain towards $0$ will also maximise the log-likelihood, while pushing down the KL-divergence until $q_\phi(\mathbf{z} | \mathbf{x}) $ cannot move closer to natures true distribution, $p_\theta(\mathbf{z} | \mathbf{x})$. So how close the lower bound can get to the log-likelihood is dependent on the flexibility of the distribution we choose for $q_\phi(\mathbf{z} | \mathbf{x}) $.



## Importance Sampling in VAEs

In VAEs, the true posterior distribution $p_\theta(\mathbf{z} | \mathbf{x})$ is unknown. Instead we use an approximate posterior $q_\theta(\mathbf{z} | \mathbf{x})$. $q$ is tractable and one can sample from it. This is an instance of importance-sampling: evaluating a quantity $\mathbb{E}_{p(\mathbf{z})} \left[  f(\mathbf{z})\right]$ given samples from another distribtion $q$.

### 1. Illustration

Let us consider a small example to illustrate importance-sampling. In this example we define a distribution $p(\mathbf{z})$ from which we (in practice) cannot sample and define a distribution $q(\mathbf{z})$ from which we can sample.

Then we can estimate $\mathbb{E}_{p(\mathbf{z})} \left[  f(\mathbf{z})\right]$ using samples from $q$:

$$\mathbb{E}_{p(\mathbf{z})} \left[  f(\mathbf{z})\right] = \mathbb{E}_{q(\mathbf{z})} \left[ w(\mathbf{z}) f(\mathbf{z})\right], \quad w(\mathbf{z}) := \frac{p(\mathbf{z})}{q(\mathbf{z})}$$

In this example, we use $f(\mathbf{z}) = \mathbf{z}$.
"""

import torch
from torch import nn
from torch.distributions import Normal
cuda = torch.cuda.is_available()

n_samples = 10000
c1 = sns.color_palette()[0]
c2 = sns.color_palette()[1]

# define p(z) and q(z)
p = Normal(loc=5 * torch.ones((1,)), scale=torch.ones((1,)))
q = Normal(loc=torch.ones((1,)), scale=3*torch.ones((1,)))

# sample z ~ q(z)
q_samples = q.sample(sample_shape=torch.Size([n_samples]))

# compute the densities p(z) and q(z)
p_prob = p.log_prob(q_samples).exp()
q_prob = q.log_prob(q_samples).exp()

# define the importance-weights: w(z) = p(z) / q(z)
w = p_prob / (q_prob + 1e-12)

# compute the importance-weighted estimate \hat{\mu} = E_q[ w(z) f(z)] where f(z) = z
estimate = (w * q_samples).mean()

# compute the ground true
p_samples = p.sample(sample_shape=torch.Size([n_samples]))
empirical_mean = p_samples.mean()

# display the results
print(f">> Empirical mean = {empirical_mean:.3f}, estimate = {estimate:.3f}")
plt.figure(figsize=(16,6))
sns.distplot(p_samples, label="$p(z)$", hist_kws={'alpha':0.2}, color=c1)
sns.distplot(q_samples, label="$q(z)$",hist_kws={'alpha':0.2}, color=c2)
plt.axvline(empirical_mean, color=c1,linestyle="-",label= "Empirical Mean")
plt.axvline(estimate, color=c2,linestyle="-", label= "Importance-Weighted Estimate")
plt.legend()
plt.show()

"""### 2. Application: Accurate Estimate of $\log p_\theta(\mathbf{x})$ using Importance Sampling

The ELBO is a lower bound to the marginal log likelihood, however it is not guaranteed to be accurate. [Importance-sampling can be leveraged in VAEs](https://arxiv.org/abs/1509.00519) to provide a more accurate estimate of the marginal log-likelihood $\log p_\theta(\mathbf{x})$ using multiple samples from $q_\phi(\mathbf{z}_1,\dots,\mathbf{z}_K | \mathbf{x})$. With $K+1$ samples $q_\phi(\mathbf{z} | \mathbf{x})$, the following inequality holds:

$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}_{K+1}(\mathcal{x}) \geq \mathcal{L}_K(\mathcal{x}) \geq \mathcal{L}(\mathcal{x})$$

where

$$ \mathcal{L}_{K}(\mathcal{x}) := \mathbb{E}_{q_\phi(\mathbf{z}_1,\dots,\mathbf{z}_K | \mathbf{x})} \left[
\log \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(\mathbf{x}, \mathbf{z}_k)}{q_\phi(\mathbf{z}_k | \mathbf{x})}
\right]$$


"""