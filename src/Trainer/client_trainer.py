import os
import logging
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ClientTrainer:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        algorithm: str = "fedavg",
        fedprox_mu: float = 0.01,
        device: str = "cpu",
        save_dir: str = "./checkpoints"
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.algorithm = algorithm.lower()
        self.fedprox_mu = fedprox_mu
        self.device = device
        self.save_dir = save_dir
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.previous_global_model = None

    def set_parameters(self, global_parameters: dict):
        """Loads global model weights into local model."""
        self.model.load_state_dict(global_parameters)
        if self.algorithm == "fedprox":
            # Save reference copy of global parameters for proximal regularization
            self.previous_global_model = copy.deepcopy(self.model)
            for param in self.previous_global_model.parameters():
                param.requires_grad = False

    def get_parameters(self) -> dict:
        """Returns local model state dict."""
        return self.model.state_dict()

    def train(self) -> float:
        """Executes local training loop over configured epochs."""
        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for epoch in range(self.epochs):
            for batch in self.train_loader:
                # Handle single Tensor vs (X, y) batch shapes
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

                self.optimizer.zero_grad()
                reconstruction = self.model(data)
                reconstruction_loss = self.criterion(reconstruction, data)

                total_loss = reconstruction_loss

                # FedProx Proximal Term Addition (Before backward pass)
                if self.algorithm == "fedprox" and self.previous_global_model is not None:
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    
                    total_loss = reconstruction_loss + (self.fedprox_mu / 2.0) * prox_term

                total_loss.backward()
                self.optimizer.step()

                running_loss += reconstruction_loss.item()
                total_batches += 1

        epoch_loss = running_loss / max(total_batches, 1)
        return epoch_loss

    def save_model(self):
        """Saves local client model to disk safely."""
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, f"client_{self.client_id}_model.cpt")
        try:
            torch.save(self.model.state_dict(), save_file, _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(self.model.state_dict(), save_file)
