"""
Variational Autoencoder model definition.

Pipeline:
    X selected features
        ↓
    32-dimensional hidden representation
        ↓
    mean (mu) and log-variance (logvar)
        ↓
    16-dimensional latent space
        ↓
    32-dimensional decoder representation
        ↓
    X reconstructed features

The model is designed for normal-only / unsupervised
anomaly detection using reconstruction error.

@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
"""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ================================================================
# VAE ENCODER
# ================================================================

class Encoder(nn.Module):
    """
    Encoder part of the Variational Autoencoder.

    Architecture:

        input_dim
            ↓
          32
            ↓
        mu --------┐
                   ├──> reparameterization -> latent z (16)
        logvar ----┘
    """

    def __init__(
        self,
        input_dim,
        hidden_neus=32,
        latent_dim=16
    ):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_neus = hidden_neus
        self.latent_dim = latent_dim

        # --------------------------------------------------------
        # Shared encoder network
        #
        # X -> 32
        # --------------------------------------------------------

        self.encoder_network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_neus,
                bias=True
            ),
            nn.ReLU()
        )

        # --------------------------------------------------------
        # Mean vector
        #
        # 32 -> 16
        # --------------------------------------------------------

        self.mu_layer = nn.Linear(
            hidden_neus,
            latent_dim,
            bias=True
        )

        # --------------------------------------------------------
        # Log variance vector
        #
        # 32 -> 16
        # --------------------------------------------------------

        self.logvar_layer = nn.Linear(
            hidden_neus,
            latent_dim,
            bias=True
        )

        self.init_params()

    def init_params(self):
        """
        Initializes all Linear layer parameters.
        """

        for layer in self.modules():

            if isinstance(
                layer,
                nn.Linear
            ):

                bound = (
                    1
                    /
                    np.sqrt(
                        layer.in_features
                    )
                )

                layer.weight.data.uniform_(
                    -bound,
                    bound
                )

                layer.bias.data.zero_()

    def forward(self, inputs):
        """
        Returns:
            mu
            logvar
        """

        hidden = self.encoder_network(
            inputs
        )

        mu = self.mu_layer(
            hidden
        )

        logvar = self.logvar_layer(
            hidden
        )

        return mu, logvar


# ================================================================
# VAE DECODER
# ================================================================

class Decoder(nn.Module):
    """
    Decoder part of the Variational Autoencoder.

    Architecture:

        latent 16
            ↓
          32
            ↓
        output_dim (= selected X features)
    """

    def __init__(
        self,
        latent_dim=16,
        hidden_neus=32,
        output_dim=None,
        use_sigmoid=False
    ):
        super(Decoder, self).__init__()

        if output_dim is None:
            raise ValueError(
                "output_dim must be provided for the Decoder."
            )

        self.latent_dim = latent_dim
        self.hidden_neus = hidden_neus
        self.output_dim = output_dim

        decoder_network = [
            nn.Linear(
                latent_dim,
                hidden_neus,
                bias=True
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_neus,
                output_dim,
                bias=True
            )
        ]

        # --------------------------------------------------------
        # Optional sigmoid
        #
        # Use when input/output is scaled to [0,1].
        # --------------------------------------------------------

        if use_sigmoid:

            decoder_network.append(
                nn.Sigmoid()
            )

        self.decoder_network = nn.Sequential(
            *decoder_network
        )

        self.init_params()

    def init_params(self):
        """
        Initializes all Linear layer parameters.
        """

        for layer in self.modules():

            if isinstance(
                layer,
                nn.Linear
            ):

                bound = (
                    1
                    /
                    np.sqrt(
                        layer.in_features
                    )
                )

                layer.weight.data.uniform_(
                    -bound,
                    bound
                )

                layer.bias.data.zero_()

    def forward(self, latent):
        """
        Decodes latent representation into
        reconstructed selected features.
        """

        return self.decoder_network(
            latent
        )


# ================================================================
# VARIATIONAL AUTOENCODER
# ================================================================

class Autoencoder(nn.Module):
    """
    Variational Autoencoder.

    Architecture:

        X
        ↓
        32
        ↓
      mu, logvar
        ↓
      z (16)
        ↓
        32
        ↓
        X_hat

    The class name "Autoencoder" is intentionally retained
    for compatibility with the existing FedMSE codebase.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_neus=32,
        latent_dim=16,
        use_sigmoid=False
    ):
        super(Autoencoder, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neus = hidden_neus
        self.latent_dim = latent_dim

        # --------------------------------------------------------
        # Encoder
        #
        # X -> 32 -> mu/logvar
        # --------------------------------------------------------

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_neus=hidden_neus,
            latent_dim=latent_dim
        )

        # --------------------------------------------------------
        # Decoder
        #
        # 16 -> 32 -> X
        # --------------------------------------------------------

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_neus=hidden_neus,
            output_dim=output_dim,
            use_sigmoid=use_sigmoid
        )

    # ============================================================
    # REPARAMETERIZATION
    # ============================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):
        """
        Reparameterization trick.

        sigma = exp(0.5 * logvar)

        z = mu + sigma * epsilon

        where:

            epsilon ~ N(0, I)
        """

        # Standard deviation
        std = torch.exp(
            0.5 * logvar
        )

        # Random Gaussian noise
        eps = torch.randn_like(
            std
        )

        # Sample latent vector
        z = (
            mu
            +
            eps * std
        )

        return z

    # ============================================================
    # KL LOSS
    # ============================================================

    def kl_loss(
        self,
        mu,
        logvar
    ):
        """
        KL divergence:

            KL(q(z|x) || N(0,I))

        Returns mean KL loss over the batch.
        """

        kl = -0.5 * torch.sum(
            1
            +
            logvar
            -
            mu.pow(2)
            -
            logvar.exp(),
            dim=1
        )

        return torch.mean(
            kl
        )

    # ============================================================
    # RECONSTRUCTION LOSS
    # ============================================================

    def recon_loss(
        self,
        input,
        output
    ):
        """
        Mean Squared Reconstruction Error.
        """

        return F.mse_loss(
            output,
            input,
            reduction='mean'
        )

    # ============================================================
    # TOTAL VAE LOSS
    # ============================================================

    def vae_loss(
        self,
        input,
        output,
        mu,
        logvar,
        beta=0.001
    ):
        """
        Total VAE loss:

            reconstruction loss
            +
            beta * KL loss

        Returns:
            total_loss
            reconstruction_loss
            kl_loss
        """

        reconstruction_loss = self.recon_loss(
            input,
            output
        )

        kl = self.kl_loss(
            mu,
            logvar
        )

        total_loss = (
            reconstruction_loss
            +
            beta * kl
        )

        return (
            total_loss,
            reconstruction_loss,
            kl
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        input
    ):
        """
        Forward pass.

        Returns:

            reconstruction
            mu
            logvar

        This format is compatible with the updated
        ClientTrainer.
        """

        # --------------------------------------------------------
        # Encoder
        # --------------------------------------------------------

        mu, logvar = self.encoder(
            input
        )

        # --------------------------------------------------------
        # Sample latent vector
        # --------------------------------------------------------

        latent = self.reparameterize(
            mu,
            logvar
        )

        # --------------------------------------------------------
        # Decoder
        # --------------------------------------------------------

        output = self.decoder(
            latent
        )

        return (
            output,
            mu,
            logvar
        )

    # ============================================================
    # ENCODE
    # ============================================================

    def encode(
        self,
        input
    ):
        """
        Returns mu, logvar.

        Useful when the latent distribution is needed.
        """

        return self.encoder(
            input
        )

    # ============================================================
    # SAMPLE LATENT
    # ============================================================

    def sample_latent(
        self,
        input
    ):
        """
        Returns a sampled latent vector z.
        """

        mu, logvar = self.encoder(
            input
        )

        return self.reparameterize(
            mu,
            logvar
        )

    # ============================================================
    # DECODE
    # ============================================================

    def decode(
        self,
        latent
    ):
        """
        Reconstructs data from latent vector.
        """

        return self.decoder(
            latent
        )

    # ============================================================
    # NUMPY HELPER
    # ============================================================

    def _to_numpy(
        self,
        tensor
    ):
        return (
            tensor.detach()
            .cpu()
            .numpy()
        )
