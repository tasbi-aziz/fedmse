"""
Normal Autoencoder model definition.

Pipeline:
    X raw features
        ↓
    32-dimensional hidden representation
        ↓
    16-dimensional latent representation
        ↓
    32-dimensional decoder representation
        ↓
    X reconstructed features

The model is designed for normal-only / unsupervised
anomaly detection using reconstruction error.

This is a standard / vanilla Autoencoder.
It does NOT use:
    - Variational encoding
    - mu / logvar
    - Reparameterization
    - KL divergence
    - Shrink penalty
    - shrink_lambda

@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
"""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ================================================================
# AE ENCODER
# ================================================================

class Encoder(nn.Module):
    """
    Encoder part of a standard Autoencoder.

    Architecture:

        input_dim
            ↓
           32
            ↓
           16
         latent
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
        # Input -> Hidden -> Latent
        # --------------------------------------------------------

        self.encoder_network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_neus,
                bias=True
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_neus,
                latent_dim,
                bias=True
            )
        )

        self.init_params()

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        inputs
    ):
        """
        Encodes input into latent representation.
        """

        return self.encoder_network(
            inputs
        )


# ================================================================
# AE DECODER
# ================================================================

class Decoder(nn.Module):
    """
    Decoder part of a standard Autoencoder.

    Architecture:

        latent 16
            ↓
           32
            ↓
        output_dim

    For the current raw-data experiment:

        16 -> 32 -> 115
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
        # Normally FALSE for the current raw-data experiment
        # because the input is NOT scaled to [0,1].
        # --------------------------------------------------------

        if use_sigmoid:

            decoder_network.append(
                nn.Sigmoid()
            )

        self.decoder_network = nn.Sequential(
            *decoder_network
        )

        self.init_params()

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        latent
    ):
        """
        Decodes latent representation into
        reconstructed features.
        """

        return self.decoder_network(
            latent
        )


# ================================================================
# NORMAL AUTOENCODER
# ================================================================

class Autoencoder(nn.Module):
    """
    Standard / Vanilla Autoencoder.

    Architecture:

        X
        ↓
       32
        ↓
       16
        ↓
       32
        ↓
       X_hat

    For the current raw-data experiment:

        115
         ↓
        32
         ↓
        16
         ↓
        32
         ↓
        115

    The class name "Autoencoder" is retained for
    compatibility with the existing FedMSE codebase.

    This model does NOT contain:

        - mu
        - logvar
        - reparameterization
        - KL divergence
        - beta
        - shrink penalty
        - shrink_lambda
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
        # input -> 32 -> 16
        # --------------------------------------------------------

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_neus=hidden_neus,
            latent_dim=latent_dim
        )

        # --------------------------------------------------------
        # Decoder
        #
        # 16 -> 32 -> output
        # --------------------------------------------------------

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_neus=hidden_neus,
            output_dim=output_dim,
            use_sigmoid=use_sigmoid
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

            output

        where output is the reconstructed input.

        This is the standard Autoencoder forward flow.
        """

        # --------------------------------------------------------
        # Encode
        # --------------------------------------------------------

        latent = self.encoder(
            input
        )

        # --------------------------------------------------------
        # Decode
        # --------------------------------------------------------

        output = self.decoder(
            latent
        )

        return output

    # ============================================================
    # ENCODE
    # ============================================================

    def encode(
        self,
        input
    ):
        """
        Returns the deterministic latent representation.
        """

        return self.encoder(
            input
        )

    # ============================================================
    # DECODE
    # ============================================================

    def decode(
        self,
        latent
    ):
        """
        Reconstructs data from latent representation.
        """

        return self.decoder(
            latent
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

        This is the main training/anomaly-detection
        loss used by the standard Autoencoder.
        """

        return F.mse_loss(
            output,
            input,
            reduction='mean'
        )

    # ============================================================
    # LOSS
    # ============================================================

    def loss(
        self,
        input,
        output
    ):
        """
        Returns the reconstruction MSE.

        No KL loss is used because this is a
        standard Autoencoder, not a VAE.
        """

        return self.recon_loss(
            input,
            output
        )

    # ============================================================
    # NUMPY HELPER
    # ============================================================

    def _to_numpy(
        self,
        tensor
    ):
        """
        Converts a PyTorch tensor to NumPy.
        """

        return (
            tensor.detach()
            .cpu()
            .numpy()
        )
