"""
This is Autoencoder model definition.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-16 20:07:29
"""

from itertools import chain
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

class Encoder(nn.Module):
    """
    Encoder module for AutoEncoder network.
    """
    def __init__(self, input_dim=115, hidden_neus=27, latent_dim=7):
        super(Encoder, self).__init__()
        encoder_network = [
            nn.Linear(input_dim, hidden_neus, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_neus, latent_dim, bias=True)
        ]
        self.encoder_network = nn.Sequential(*encoder_network)
        self.init_params()

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        return self.encoder_network(inputs)


class Decoder(nn.Module):
    """
    Decoder module for AutoEncoder network.
    """
    def __init__(self, latent_dim=7, hidden_neus=27, output_dim=115):
        super(Decoder, self).__init__()
        decoder_network = [
            nn.Linear(latent_dim, hidden_neus, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_neus, output_dim, bias=True)
        ]
        self.decoder_network = nn.Sequential(*decoder_network)
        self.init_params()

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, latent):
        return self.decoder_network(latent)


class Autoencoder(nn.Module):
    """
    Autoencoder class
    """
    def __init__(self, input_dim=115, output_dim=115, hidden_neus=27, latent_dim=7):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)

    def parameters(self, recurse: bool = True):
        return chain(self.encoder.parameters(recurse=recurse), self.decoder.parameters(recurse=recurse))

    def recon_loss(self, input, output):
        return F.mse_loss(input, output, reduction='mean')
    
    def forward(self, input):
        latent = self.encoder(input)
        output = self.decoder(latent)
        loss = self.recon_loss(input, output)
        return latent, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()
