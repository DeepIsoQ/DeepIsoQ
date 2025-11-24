import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import Image, display, clear_output
from sklearn.manifold import TSNE
from torch import Tensor
from torch.distributions import Normal
from torchvision.utils import make_grid


def plot_autoencoder_stats(
        x: Tensor = None,
        x_hat: Tensor = None,
        z: Tensor = None,
        y: Tensor = None,
        epoch: int = None,
        train_loss: List = None,
        valid_loss: List = None,
        classes: List = None,
        dimensionality_reduction_op: Optional[Callable] = None,
) -> None:
    """
    An utility 
    """
    # -- Plotting --
    f, axarr = plt.subplots(2, 2, figsize=(20, 20))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    ax.plot(np.arange(epoch + 1), train_loss, color="black")
    ax.plot(np.arange(epoch + 1), valid_loss, color="gray", linestyle="--")
    ax.legend(['Training error', 'Validation error'])

    # Latent space
    ax = axarr[0, 1]

    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # If you want to use a dimensionality reduction method you can use
    # for example TSNE by projecting on two principal dimensions
    # TSNE.fit_transform(z)
    if dimensionality_reduction_op is not None:
        z = dimensionality_reduction_op(z)

    colors = iter(plt.get_cmap('Set1')(np.linspace(0, 1.0, len(classes))))
    for c in classes:
        ax.scatter(*z[y.numpy() == c].T, c=next(colors), marker='o')

    ax.legend(classes)

def make_vae_plots(
    vae,
    x,
    outputs,
    training_data,
    validation_data,
    tmp_img="tmp_vae_out.png",
    figsize=(12, 4),
):
    """
    Plot only:
      - ELBO
      - KL
      - log p(x|z)  (NLL term)
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    ax_elbo, ax_kl, ax_nll = axes[0]

    # ---- ELBO ----
    ax_elbo.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    ax_elbo.plot(training_data['elbo'], label='Training')
    ax_elbo.plot(validation_data['elbo'], label='Validation')
    ax_elbo.set_xlabel('Epoch')
    ax_elbo.legend()

    # ---- KL ----
    ax_kl.set_title(
        r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$'
    )
    ax_kl.plot(training_data['kl'], label='Training')
    ax_kl.plot(validation_data['kl'], label='Validation')
    ax_kl.set_xlabel('Epoch')
    ax_kl.legend()

    # ---- log p(x|z) ----
    ax_nll.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
    ax_nll.plot(training_data['log_px'], label='Training')
    ax_nll.plot(validation_data['log_px'], label='Validation')
    ax_nll.set_xlabel('Epoch')
    ax_nll.legend()

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)
