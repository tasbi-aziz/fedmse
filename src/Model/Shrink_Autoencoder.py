import torch
import torch.nn as nn
from itertools import chain

class Shrink_Autoencoder(nn.Module):
    def __init__(self, input_dim: int, shrink_dim: int = 16, threshold: float = 0.2):
        super(Shrink_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.shrink_dim = shrink_dim
        self.threshold = threshold

        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.shrink_dim)
        )

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(self.shrink_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim)
        )

    def shrinkage_operator(self, z: torch.Tensor) -> torch.Tensor:
        """
        Soft-thresholding / Shrinkage Operator:
        Suppresses small activation noise in the latent space (z) while maintaining autograd graph.
        Formula: sign(z) * max(|z| - threshold, 0)
        """
        return torch.sign(z) * torch.relu(torch.abs(z) - self.threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through encoder to extract latent representation
        z = self.encoder(x)
        
        # Apply shrinkage function to filter noise in bottleneck
        z_shrunk = self.shrinkage_operator(z)
        
        # Reconstruct input via decoder
        reconstructed = self.decoder(z_shrunk)
        return reconstructed

    # Fix: Renamed method to parameters (fixed spelling)
    def parameters(self, recurse: bool = True):
        return chain(self.encoder.parameters(recurse=recurse), self.decoder.parameters(recurse=recurse))
