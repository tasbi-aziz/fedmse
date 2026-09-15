import os
import time
import logging
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ClientTrainer:
    def __init__(
        self,
        model: nn.Module,
        client_id: int = 0,
        train_loader: DataLoader = None,
        epochs: int = 5,
        epoch: int = None,            # Compatibility alias
        lr: float = 0.001,
        lr_rate: float = None,        # Compatibility alias
        algorithm: str = "fedavg",
        update_type: str = None,      # Compatibility alias
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
        
        # Tracked metrics for Security Buffer & Aggregator
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_loss_variance = 0.0
        self.sub_sample_losses = []   # Raw sub-sample losses
        self.val_mse_list = []        # 5-fold MSE Behavioral Signature list
        self.train_time = 0.0         # Measured local training execution time (seconds)
        self.dataset_size = 0         # Local training dataset size (n_i)

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

    def _get_reconstruction(self, output_obj, data=None):
        """Helper to extract reconstruction tensor reliably based on input shape."""
        if isinstance(output_obj, (tuple, list)):
            if data is not None:
                for item in output_obj:
                    if isinstance(item, torch.Tensor) and item.shape == data.shape:
                        return item
            return output_obj[0]
        return output_obj

    def train(self, train_loader: DataLoader = None) -> float:
        """Executes local training loop over configured epochs and records timing & dataset size."""
        loader = train_loader if train_loader is not None else self.train_loader
        if loader is None:
            raise ValueError("No train_loader provided to ClientTrainer.")

        self.model.train()
        running_loss = 0.0
        total_batches = 0

        # Track dataset size (n_i)
        self.dataset_size = len(loader.dataset) if loader.dataset is not None else 0
        start_time = time.time()

        for ep in range(self.epochs):
            for batch in loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

                self.optimizer.zero_grad()
                output_obj = self.model(data)
                reconstruction = self._get_reconstruction(output_obj, data)
                
                reconstruction_loss = self.criterion(reconstruction, data)
                total_loss = reconstruction_loss

                # FedProx Proximal Term
                if self.algorithm == "fedprox" and self.previous_global_model is not None:
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    
                    total_loss = reconstruction_loss + (self.fedprox_mu / 2.0) * prox_term

                total_loss.backward()

                # Gradient Clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += reconstruction_loss.item()
                total_batches += 1

        self.train_time = time.time() - start_time
        self.train_loss = running_loss / max(total_batches, 1)
        return self.train_loss

    def evaluate(self, valid_loader: DataLoader, num_folds: int = 5) -> tuple:
        """Evaluates local model on validation set using a 5-fold sub-sampling approach."""
        if valid_loader is None or valid_loader.dataset is None or len(valid_loader.dataset) == 0:
            self.val_loss = self.train_loss
            self.val_loss_variance = 0.0
            self.val_mse_list = [self.train_loss] * num_folds
            self.sub_sample_losses = self.val_mse_list
            return self.val_loss, self.val_loss_variance

        self.model.eval()
        dataset = valid_loader.dataset
        total_size = len(dataset)
        
        folds = min(num_folds, total_size)
        fold_size = total_size // folds
        
        sub_losses = []

        with torch.no_grad():
            for i in range(folds):
                start_idx = i * fold_size
                end_idx = total_size if i == folds - 1 else (i + 1) * fold_size
                indices = list(range(start_idx, end_idx))
                
                sub_dataset = Subset(dataset, indices)
                sub_loader = DataLoader(
                    sub_dataset,
                    batch_size=valid_loader.batch_size or 32,
                    shuffle=False
                )

                fold_loss = 0.0
                total_batches = 0

                for batch in sub_loader:
                    data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                    output_obj = self.model(data)
                    reconstruction = self._get_reconstruction(output_obj, data)
                    
                    loss = self.criterion(reconstruction, data)
                    fold_loss += loss.item()
                    total_batches += 1

                avg_fold_loss = fold_loss / max(total_batches, 1)
                sub_losses.append(avg_fold_loss)

        # Ensure exactly num_folds (5) entries in the MSE list
        while len(sub_losses) < num_folds:
            sub_losses.append(sub_losses[-1] if sub_losses else 0.0)

        loss_tensor = torch.tensor(sub_losses, dtype=torch.float32)
        self.sub_sample_losses = sub_losses[:num_folds]
        self.val_mse_list = self.sub_sample_losses
        self.val_loss = torch.mean(loss_tensor).item()
        self.val_loss_variance = torch.var(loss_tensor, unbiased=False).item() if len(sub_losses) > 1 else 0.0

        return self.val_loss, self.val_loss_variance

    def run(self, train_loader: DataLoader, valid_loader: DataLoader = None) -> tuple:
        """Pipeline runner: executes training, 5-fold sub-sample validation, and auto-saves model."""
        self.train(train_loader)
        if valid_loader is not None:
            val_loss, val_variance = self.evaluate(valid_loader, num_folds=5)
        else:
            val_loss, val_variance = self.train_loss, 0.0

        self.save_model()
        return val_loss, val_variance

    def get_update_payload() -> dict:
        """Helper to collect client update payload for Security Buffer."""
        return {
            "client_id": self.client_id,
            "weights": self.get_parameters(),
            "val_mse_list": self.val_mse_list,
            "dataset_size": self.dataset_size,
            "train_time": self.train_time,
            "val_loss": self.val_loss,
            "val_variance": self.val_loss_variance,
            "train_loss": self.train_loss
        }

    def save_model(self):
        """Saves local client model to disk safely."""
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, f"client_{self.client_id}_model.cpt")
        try:
            torch.save(self.model.state_dict(), save_file, _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(self.model.state_dict(), save_file)
