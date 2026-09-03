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
        model: nn.Module,
        client_id: int = 0,
        train_loader: DataLoader = None,
        epochs: int = 5,
        epoch: int = None,            # Compatibility alias for epoch
        lr: float = 0.001,
        lr_rate: float = None,        # Compatibility alias for lr_rate
        algorithm: str = "fedavg",
        update_type: str = None,      # Compatibility alias for update_type
        fedprox_mu: float = 0.01,
        device: str = "cpu",
        save_dir: str = "./checkpoints"
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.epochs = epoch if epoch is not None else epochs
        self.lr = lr_rate if lr_rate is not None else lr
        
        algo_choice = update_type if update_type is not None else algorithm
        self.algorithm = algo_choice.lower()
        
        self.fedprox_mu = fedprox_mu
        self.device = device
        self.save_dir = save_dir
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.previous_global_model = None
        
        # Tracked loss metrics
        self.train_loss = 0.0
        self.val_loss = 0.0

        if self.algorithm == "fedprox":
            self.previous_global_model = copy.deepcopy(self.model)
            for param in self.previous_global_model.parameters():
                param.requires_grad = False

    def set_parameters(self, global_parameters: dict):
        """Loads global model weights into local model."""
        self.model.load_state_dict(global_parameters)
        if self.algorithm == "fedprox":
            self.previous_global_model = copy.deepcopy(self.model)
            for param in self.previous_global_model.parameters():
                param.requires_grad = False

    def get_parameters(self) -> dict:
        """Returns local model state dict."""
        return self.model.state_dict()

    def _get_reconstruction(self, output_obj):
        """Helper to extract reconstruction tensor if model output is a tuple."""
        if isinstance(output_obj, (tuple, list)):
            return output_obj[1]  # Extract output from (latent, output, loss)
        return output_obj

    def train(self, train_loader: DataLoader = None) -> float:
        """Executes local training loop over configured epochs."""
        loader = train_loader if train_loader is not None else self.train_loader
        if loader is None:
            raise ValueError("No train_loader provided to ClientTrainer.")

        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for ep in range(self.epochs):
            for batch in loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

                self.optimizer.zero_grad()
                output_obj = self.model(data)
                reconstruction = self._get_reconstruction(output_obj)
                
                reconstruction_loss = self.criterion(reconstruction, data)

                total_loss = reconstruction_loss

                # FedProx Proximal Term
                if self.algorithm == "fedprox" and self.previous_global_model is not None:
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    
                    total_loss = reconstruction_loss + (self.fedprox_mu / 2.0) * prox_term

                total_loss.backward()
                self.optimizer.step()

                running_loss += reconstruction_loss.item()
                total_batches += 1

        self.train_loss = running_loss / max(total_batches, 1)
        return self.train_loss

    def evaluate(self, valid_loader: DataLoader) -> float:
        """Evaluates local model on validation set."""
        if valid_loader is None:
            self.val_loss = self.train_loss
            return self.val_loss

        self.model.eval()
        running_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                output_obj = self.model(data)
                reconstruction = self._get_reconstruction(output_obj)
                
                loss = self.criterion(reconstruction, data)
                running_loss += loss.item()
                total_batches += 1

        self.val_loss = running_loss / max(total_batches, 1)
        return self.val_loss

    def run(self, train_loader: DataLoader, valid_loader: DataLoader = None) -> float:
        """Pipeline runner: executes training, validation, and auto-saves model."""
        self.train(train_loader)
        val_loss = self.evaluate(valid_loader) if valid_loader is not None else self.train_loss
        self.save_model()
        return val_loss

    def save_model(self):
        """Saves local client model to disk safely."""
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, f"client_{self.client_id}_model.cpt")
        try:
            torch.save(self.model.state_dict(), save_file, _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(self.model.state_dict(), save_file)
