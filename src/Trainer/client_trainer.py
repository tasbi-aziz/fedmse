import os
import time
import logging
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClientTrainer:

    def __init__(
        self,
        model: nn.Module,
        client_id: int = 0,
        train_loader: DataLoader = None,
        epochs: int = 5,
        epoch: int = None,              # Compatibility alias
        lr: float = 0.001,
        lr_rate: float = None,          # Compatibility alias
        algorithm: str = "fedavg",
        update_type: str = None,        # Compatibility alias
        fedprox_mu: float = 0.01,
        device: str = "cpu",
        save_dir: str = "./checkpoints",
        kl_weight: float = 0.001        # VAE KL-divergence weight
    ):

        self.client_id = client_id

        self.model = copy.deepcopy(
            model
        ).to(device)

        self.train_loader = train_loader

        self.epochs = (
            epoch if epoch is not None else epochs
        )

        self.lr = (
            lr_rate if lr_rate is not None else lr
        )

        algo_choice = (
            update_type
            if update_type is not None
            else algorithm
        )

        self.algorithm = algo_choice.lower()

        self.fedprox_mu = fedprox_mu
        self.device = device
        self.save_dir = save_dir

        # ---------------------------------------------------------
        # VAE KL-divergence weight
        # ---------------------------------------------------------
        self.kl_weight = kl_weight

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        # Reconstruction loss
        self.criterion = nn.MSELoss()

        self.previous_global_model = None

        # ---------------------------------------------------------
        # Tracked metrics for Security Buffer & Aggregator
        # ---------------------------------------------------------

        self.train_loss = 0.0

        self.val_loss = 0.0

        self.val_loss_variance = 0.0

        self.sub_sample_losses = []

        # 5-fold MSE behavioral signature
        self.val_mse_list = []

        # Actual local training execution time
        self.train_time = 0.0

        # Local training dataset size
        self.dataset_size = 0

        # Optional diagnostics
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0

        # ---------------------------------------------------------
        # FedProx
        # ---------------------------------------------------------

        if self.algorithm == "fedprox":

            self.previous_global_model = copy.deepcopy(
                self.model
            )

            for param in self.previous_global_model.parameters():
                param.requires_grad = False

    # =============================================================
    # SET PARAMETERS
    # =============================================================

    def set_parameters(self, global_parameters: dict):
        """
        Loads global model weights into local model.
        """

        self.model.load_state_dict(
            global_parameters
        )

        if self.algorithm == "fedprox":

            self.previous_global_model = copy.deepcopy(
                self.model
            )

            for param in self.previous_global_model.parameters():
                param.requires_grad = False

    # =============================================================
    # GET PARAMETERS
    # =============================================================

    def get_parameters(self) -> dict:
        """
        Returns local model state dict.
        """

        return self.model.state_dict()

    # =============================================================
    # EXTRACT RECONSTRUCTION
    # =============================================================

    def _get_reconstruction(
        self,
        output_obj,
        data=None
    ):
        """
        Extracts reconstruction tensor from model output.

        Supported outputs:

        AE:
            reconstruction

        VAE:
            reconstruction, mu, logvar
        """

        if isinstance(
            output_obj,
            (tuple, list)
        ):

            if data is not None:

                for item in output_obj:

                    if (
                        isinstance(item, torch.Tensor)
                        and item.shape == data.shape
                    ):
                        return item

            return output_obj[0]

        return output_obj

    # =============================================================
    # EXTRACT VAE PARAMETERS
    # =============================================================

    def _get_vae_parameters(
        self,
        output_obj
    ):
        """
        Extracts mu and logvar from a VAE output.

        Expected VAE output:

            reconstruction, mu, logvar

        If the model is a normal AE, returns None, None.
        """

        if isinstance(
            output_obj,
            (tuple, list)
        ):

            # Typical VAE output:
            # output[0] = reconstruction
            # output[1] = mu
            # output[2] = logvar

            if len(output_obj) >= 3:

                mu = output_obj[1]
                logvar = output_obj[2]

                if (
                    isinstance(mu, torch.Tensor)
                    and isinstance(logvar, torch.Tensor)
                ):
                    return mu, logvar

        return None, None

    # =============================================================
    # VAE KL LOSS
    # =============================================================

    def _compute_kl_loss(
        self,
        mu,
        logvar
    ):
        """
        Computes KL divergence between:

            q(z|x) = N(mu, sigma^2)

        and:

            p(z) = N(0, I)

        Formula:

            KL = -0.5 * sum(
                1 + logvar - mu^2 - exp(logvar)
            )

        Averaged over the batch.
        """

        if (
            mu is None
            or logvar is None
        ):
            return torch.tensor(
                0.0,
                device=self.device
            )

        kl_loss = -0.5 * torch.sum(
            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        # Average over batch
        kl_loss = kl_loss / mu.size(0)

        return kl_loss

    # =============================================================
    # TRAIN
    # =============================================================

    def train(
        self,
        train_loader: DataLoader = None
    ) -> float:

        """
        Executes local training.

        For VAE:

            total_loss =
                reconstruction_loss
                +
                kl_weight * KL_loss

        Gradient clipping is applied before optimizer.step().
        """

        loader = (
            train_loader
            if train_loader is not None
            else self.train_loader
        )

        if loader is None:
            raise ValueError(
                "No train_loader provided to ClientTrainer."
            )

        self.model.train()

        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0

        total_batches = 0

        # ---------------------------------------------------------
        # Track dataset size
        # ---------------------------------------------------------

        self.dataset_size = (
            len(loader.dataset)
            if loader.dataset is not None
            else 0
        )

        # ---------------------------------------------------------
        # Start timing
        # ---------------------------------------------------------

        start_time = time.time()

        # ---------------------------------------------------------
        # Local epochs
        # ---------------------------------------------------------

        for ep in range(self.epochs):

            for batch in loader:

                data = (
                    batch[0].to(self.device)
                    if isinstance(
                        batch,
                        (list, tuple)
                    )
                    else batch.to(self.device)
                )

                # -------------------------------------------------
                # Reset gradients
                # -------------------------------------------------

                self.optimizer.zero_grad()

                # -------------------------------------------------
                # Forward
                # -------------------------------------------------

                output_obj = self.model(
                    data
                )

                # -------------------------------------------------
                # Reconstruction
                # -------------------------------------------------

                reconstruction = self._get_reconstruction(
                    output_obj,
                    data
                )

                reconstruction_loss = self.criterion(
                    reconstruction,
                    data
                )

                # -------------------------------------------------
                # VAE parameters
                # -------------------------------------------------

                mu, logvar = self._get_vae_parameters(
                    output_obj
                )

                kl_loss = self._compute_kl_loss(
                    mu,
                    logvar
                )

                # -------------------------------------------------
                # Total VAE loss
                # -------------------------------------------------

                total_loss = (
                    reconstruction_loss
                    +
                    self.kl_weight * kl_loss
                )

                # -------------------------------------------------
                # FedProx proximal term
                # -------------------------------------------------

                if (
                    self.algorithm == "fedprox"
                    and
                    self.previous_global_model is not None
                ):

                    prox_term = 0.0

                    for (
                        param,
                        global_param
                    ) in zip(
                        self.model.parameters(),
                        self.previous_global_model.parameters()
                    ):

                        prox_term += torch.sum(
                            torch.square(
                                param
                                -
                                global_param.to(
                                    self.device
                                )
                            )
                        )

                    total_loss = (
                        total_loss
                        +
                        (self.fedprox_mu / 2.0)
                        * prox_term
                    )

                # -------------------------------------------------
                # Backward
                # -------------------------------------------------

                total_loss.backward()

                # -------------------------------------------------
                # Gradient clipping
                #
                # Prevents extremely large gradients from causing
                # unstable weight updates.
                # -------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                # -------------------------------------------------
                # Optimizer update
                # -------------------------------------------------

                self.optimizer.step()

                # -------------------------------------------------
                # Metrics
                #
                # For training loss we track reconstruction loss.
                # This keeps the existing anomaly-detection logic.
                # -------------------------------------------------

                running_loss += (
                    reconstruction_loss.item()
                )

                running_reconstruction_loss += (
                    reconstruction_loss.item()
                )

                running_kl_loss += (
                    kl_loss.item()
                )

                total_batches += 1

        # ---------------------------------------------------------
        # End timing
        # ---------------------------------------------------------

        self.train_time = (
            time.time()
            -
            start_time
        )

        # ---------------------------------------------------------
        # Final training metrics
        # ---------------------------------------------------------

        denominator = max(
            total_batches,
            1
        )

        self.train_loss = (
            running_loss
            /
            denominator
        )

        self.reconstruction_loss = (
            running_reconstruction_loss
            /
            denominator
        )

        self.kl_loss = (
            running_kl_loss
            /
            denominator
        )

        logging.info(
            f"[Client {self.client_id}] "
            f"Training complete | "
            f"Reconstruction Loss: "
            f"{self.reconstruction_loss:.6f} | "
            f"KL Loss: "
            f"{self.kl_loss:.6f} | "
            f"Time: "
            f"{self.train_time:.2f}s | "
            f"Dataset Size: "
            f"{self.dataset_size}"
        )

        return self.train_loss

    # =============================================================
    # EVALUATE
    # =============================================================

    def evaluate(
        self,
        valid_loader: DataLoader,
        num_folds: int = 5
    ) -> tuple:

        """
        Evaluates local model using 5-fold validation.

        IMPORTANT:
        Only reconstruction MSE is recorded.

        KL loss is NOT included in val_mse_list because the
        MSE list is being used as the anomaly-behavior signature.
        """

        if (
            valid_loader is None
            or valid_loader.dataset is None
            or len(valid_loader.dataset) == 0
        ):

            self.val_loss = self.train_loss

            self.val_loss_variance = 0.0

            self.val_mse_list = [
                self.train_loss
            ] * num_folds

            self.sub_sample_losses = (
                self.val_mse_list
            )

            return (
                self.val_loss,
                self.val_loss_variance
            )

        self.model.eval()

        dataset = valid_loader.dataset

        total_size = len(dataset)

        folds = min(
            num_folds,
            total_size
        )

        fold_size = (
            total_size // folds
        )

        sub_losses = []

        # ---------------------------------------------------------
        # No gradient during validation
        # ---------------------------------------------------------

        with torch.no_grad():

            for i in range(folds):

                start_idx = (
                    i * fold_size
                )

                end_idx = (
                    total_size
                    if i == folds - 1
                    else (i + 1) * fold_size
                )

                indices = list(
                    range(
                        start_idx,
                        end_idx
                    )
                )

                sub_dataset = Subset(
                    dataset,
                    indices
                )

                sub_loader = DataLoader(
                    sub_dataset,
                    batch_size=(
                        valid_loader.batch_size
                        or 32
                    ),
                    shuffle=False
                )

                fold_loss = 0.0

                total_batches = 0

                for batch in sub_loader:

                    data = (
                        batch[0].to(self.device)
                        if isinstance(
                            batch,
                            (list, tuple)
                        )
                        else batch.to(self.device)
                    )

                    output_obj = self.model(
                        data
                    )

                    reconstruction = (
                        self._get_reconstruction(
                            output_obj,
                            data
                        )
                    )

                    # -------------------------------------------------
                    # Reconstruction MSE only
                    # -------------------------------------------------

                    loss = self.criterion(
                        reconstruction,
                        data
                    )

                    fold_loss += loss.item()

                    total_batches += 1

                avg_fold_loss = (
                    fold_loss
                    /
                    max(
                        total_batches,
                        1
                    )
                )

                sub_losses.append(
                    avg_fold_loss
                )

        # ---------------------------------------------------------
        # Ensure exactly 5 entries
        # ---------------------------------------------------------

        while len(sub_losses) < num_folds:

            sub_losses.append(
                sub_losses[-1]
                if sub_losses
                else 0.0
            )

        # ---------------------------------------------------------
        # Validation metrics
        # ---------------------------------------------------------

        loss_tensor = torch.tensor(
            sub_losses,
            dtype=torch.float32
        )

        self.sub_sample_losses = (
            sub_losses[:num_folds]
        )

        self.val_mse_list = (
            self.sub_sample_losses
        )

        self.val_loss = (
            torch.mean(
                loss_tensor
            ).item()
        )

        self.val_loss_variance = (
            torch.var(
                loss_tensor,
                unbiased=False
            ).item()
            if len(sub_losses) > 1
            else 0.0
        )

        return (
            self.val_loss,
            self.val_loss_variance
        )

    # =============================================================
    # RUN
    # =============================================================

    def run(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader = None
    ) -> tuple:

        """
        Executes:

            Local VAE training
                ↓
            5-fold validation
                ↓
            Save model
        """

        self.train(
            train_loader
        )

        if valid_loader is not None:

            val_loss, val_variance = (
                self.evaluate(
                    valid_loader,
                    num_folds=5
                )
            )

        else:

            val_loss = self.train_loss
            val_variance = 0.0

        self.save_model()

        return (
            val_loss,
            val_variance
        )

    # =============================================================
    # UPDATE PAYLOAD
    # =============================================================

    def get_update_payload(self) -> dict:
        """
        Collects client update information for the Security Buffer.
        """

        return {
            "client_id": self.client_id,

            # Local model weights
            "weights": self.get_parameters(),

            # 5-fold reconstruction MSE signature
            "val_mse_list": self.val_mse_list,

            # Local workload information
            "dataset_size": self.dataset_size,

            # Actual local training execution time
            "train_time": self.train_time,

            # Average validation reconstruction MSE
            "val_loss": self.val_loss,

            # Variance among the 5 MSE values
            "val_variance": self.val_loss_variance,

            # Average reconstruction loss during training
            "train_loss": self.train_loss,

            # Optional VAE diagnostics
            "reconstruction_loss": self.reconstruction_loss,

            "kl_loss": self.kl_loss,

            "kl_weight": self.kl_weight
        }

    # =============================================================
    # SAVE MODEL
    # =============================================================

    def save_model(self):
        """
        Saves local client model to disk safely.
        """

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        save_file = os.path.join(
            self.save_dir,
            f"client_{self.client_id}_model.cpt"
        )

        try:

            torch.save(
                self.model.state_dict(),
                save_file,
                _use_new_zipfile_serialization=False
            )

        except Exception:

            torch.save(
                self.model.state_dict(),
                save_file
            )
