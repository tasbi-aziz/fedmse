import copy
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class GlobalAggregator:
    def __init__(self, global_model: nn.Module, dev_loader: DataLoader, device: str = "cpu"):
        self.global_model = global_model.to(device)
        self.dev_loader = dev_loader
        self.device = device
        self.criterion = nn.MSELoss()

    def get_global_parameters(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_weights: list, client_sizes: list):
        """
        Executes weighted FedAvg parameter aggregation.
        """
        total_samples = sum(client_sizes)
        aggregated_dict = copy.deepcopy(client_weights[0])

        # Initialize weights tensor to zeros
        for key in aggregated_dict.keys():
            aggregated_dict[key] = torch.zeros_like(aggregated_dict[key], dtype=torch.float32)

        # Weighted summation
        for i in range(len(client_weights)):
            weight_factor = client_sizes[i] / total_samples
            for key in aggregated_dict.keys():
                aggregated_dict[key] += client_weights[i][key].to(self.device) * weight_factor

        self.global_model.load_state_dict(aggregated_dict)

    def evaluate(self) -> float:
        """
        Evaluates the current global model on the validation/dev set.
        """
        self.global_model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.dev_loader:
                # IoTDataset returns (X, y) tuple
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                
                reconstruction = self.global_model(data)
                loss = self.criterion(reconstruction, data)
                
                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)
