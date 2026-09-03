import copy
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class GlobalAggregator:
    def __init__(self, global_model: nn.Module, dev_loader: DataLoader = None, device: str = "cpu", eps: float = 1e-8):
        self.global_model = global_model.to(device)
        self.dev_loader = dev_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.val_loss = float("inf")
        self.eps = eps

    @property
    def model(self):
        return self.global_model

    def get_global_parameters(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def set_dev_loader(self, dev_loader: DataLoader):
        self.dev_loader = dev_loader

    def compute_client_mse(self, client_state_dict: dict, client_val_loader: DataLoader) -> float:
        """
        Calculates MSE loss of a specific client's state_dict on its validation dataset.
        """
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(client_state_dict)
        temp_model.eval()
        temp_model.to(self.device)

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in client_val_loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction = temp_model(data)
                loss = self.criterion(reconstruction, data)
                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def aggregate_mse(self, client_weights: list, client_mses: list):
        """
        Executes MSE-weighted parameter aggregation (MSEAvg).
        Lower MSE yields higher aggregation weight: weight_i = (1 / (mse_i + eps)) / sum(1 / (mse_k + eps))
        """
        inv_mses = [1.0 / (mse + self.eps) for mse in client_mses]
        sum_inv_mses = sum(inv_mses)
        weights_factor = [inv_m / sum_inv_mses for inv_m in inv_mses]

        aggregated_dict = copy.deepcopy(client_weights[0])

        for key in aggregated_dict.keys():
            aggregated_dict[key] = torch.zeros_like(aggregated_dict[key], dtype=torch.float32)

        for i in range(len(client_weights)):
            factor = weights_factor[i]
            for key in aggregated_dict.keys():
                aggregated_dict[key] += client_weights[i][key].to(self.device) * factor

        self.global_model.load_state_dict(aggregated_dict)

    def update(self, local_models: list, client_val_loaders: list = None):
        """
        Extracts weights and computes MSE-based weights for aggregation.
        Expects local_models entries as (state_dict, total_samples, sample_count).
        """
        if not local_models:
            return

        client_weights = [m[0] for m in local_models]

        if client_val_loaders and len(client_val_loaders) == len(client_weights):
            client_mses = [
                self.compute_client_mse(w, loader) 
                for w, loader in zip(client_weights, client_val_loaders)
            ]
        else:
            # Fallback: compute MSE for each model on global dev_loader if specific client loaders aren't passed
            client_mses = [
                self.compute_client_mse(w, self.dev_loader) 
                for w in client_weights
            ]

        self.aggregate_mse(client_weights, client_mses)

        if self.dev_loader is not None:
            self.val_loss = self.evaluate()

    def evaluate(self) -> float:
        if self.dev_loader is None:
            logging.warning("dev_loader is not set. Skipping evaluation.")
            return float("inf")

        self.global_model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.dev_loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction = self.global_model(data)
                loss = self.criterion(reconstruction, data)
                total_loss += loss.item()
                total_batches += 1

        self.val_loss = total_loss / max(total_batches, 1)
        return self.val_loss
