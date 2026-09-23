"""
Overfitting Test for VAE-based Anomaly Detection

Purpose:
    Check whether the VAE overfits during local training.

Pipeline:
    Client normal data
        ↓
    Train / Validation split
        ↓
    Same preprocessing as main.py
        ↓
    VAE training
        ↓
    Epoch-wise Train Reconstruction MSE
    Epoch-wise Validation Reconstruction MSE
        ↓
    Plot Train vs Validation Reconstruction MSE
"""

import os
import sys
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Project path
# ------------------------------------------------------------------

PROJECT_ROOT = "/content/fedmse"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

from Model.autoencoder import Autoencoder
from DataLoader.data_processor import IoTDataProcessor
from Dataloader.dataset import IoTDataset


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "Configuration",
    "scen2-nba-iot-10clients.json"
)

DATA_SEED = 1234

NETWORK_SIZE = 10

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 1e-5

LATENT_DIM = 16

HIDDEN_DIM = 32

TARGET_NUM_FEATURES = 64

BOOTSTRAP_FRACTION = 0.10

VAE_KL_WEIGHT = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "overfitting_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed=1234):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# VAE loss
# ------------------------------------------------------------------

def compute_kl_loss(mu, logvar):

    logvar = torch.clamp(
        logvar,
        min=-10.0,
        max=10.0
    )

    kl = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl.sum(dim=1).mean()


# ------------------------------------------------------------------
# Get reconstruction from VAE output
# ------------------------------------------------------------------

def get_reconstruction(output):

    if isinstance(output, (tuple, list)):

        for item in output:

            if torch.is_tensor(item):
                return item

    return output


# ------------------------------------------------------------------
# Extract VAE parameters
# ------------------------------------------------------------------

def get_vae_parameters(output):

    if isinstance(output, (tuple, list)):

        tensors = [
            x for x in output
            if torch.is_tensor(x)
        ]

        if len(tensors) >= 3:

            reconstruction = tensors[0]
            mu = tensors[1]
            logvar = tensors[2]

            return reconstruction, mu, logvar

    raise ValueError(
        "VAE output does not contain reconstruction, mu and logvar."
    )


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def evaluate_validation(model, validation_loader):

    model.eval()

    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in validation_loader:

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(DEVICE).float()

            output = model(x)

            reconstruction, mu, logvar = get_vae_parameters(output)

            reconstruction_loss = F.mse_loss(
                reconstruction,
                x,
                reduction="sum"
            )

            kl_loss = compute_kl_loss(
                mu,
                logvar
            )

            batch_size = x.size(0)

            total_reconstruction_loss += (
                reconstruction_loss.item()
            )

            total_kl_loss += (
                kl_loss.item() * batch_size
            )

            total_samples += batch_size

    mean_reconstruction_loss = (
        total_reconstruction_loss / total_samples
    )

    mean_kl_loss = (
        total_kl_loss / total_samples
    )

    total_loss = (
        mean_reconstruction_loss
        + VAE_KL_WEIGHT * mean_kl_loss
    )

    return (
        mean_reconstruction_loss,
        mean_kl_loss,
        total_loss
    )


# ------------------------------------------------------------------
# One training epoch
# ------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer):

    model.train()

    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    for batch in train_loader:

        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = x.to(DEVICE).float()

        optimizer.zero_grad()

        output = model(x)

        reconstruction, mu, logvar = get_vae_parameters(
            output
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
            reduction="mean"
        )

        kl_loss = compute_kl_loss(
            mu,
            logvar
        )

        total_loss = (
            reconstruction_loss
            + VAE_KL_WEIGHT * kl_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = x.size(0)

        total_reconstruction_loss += (
            reconstruction_loss.item() * batch_size
        )

        total_kl_loss += (
            kl_loss.item() * batch_size
        )

        total_samples += batch_size

    mean_reconstruction_loss = (
        total_reconstruction_loss / total_samples
    )

    mean_kl_loss = (
        total_kl_loss / total_samples
    )

    total_loss = (
        mean_reconstruction_loss
        + VAE_KL_WEIGHT * mean_kl_loss
    )

    return (
        mean_reconstruction_loss,
        mean_kl_loss,
        total_loss
    )


# ------------------------------------------------------------------
# Load client data
# ------------------------------------------------------------------

def load_client_data():

    import json

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    devices = config["devices_list"]

    random.seed(DATA_SEED)

    selected_clients = random.sample(
        devices,
        NETWORK_SIZE
    )

    logger.info(
        "Selected %d clients.",
        len(selected_clients)
    )

    client_data = []

    for client_id in selected_clients:

        # ----------------------------------------------------------
        # Adjust these paths only if your existing main.py uses
        # a different data-loading function/path.
        # ----------------------------------------------------------

        train_path = client_id["train_data"]
        abnormal_path = client_id.get("abnormal_data", None)

        normal_data = pd.read_csv(train_path)

        normal_data = normal_data.sample(
            frac=1,
            random_state=DATA_SEED
        ).reset_index(drop=True)

        # ----------------------------------------------------------
        # Same split as main.py
        # ----------------------------------------------------------

        total_normal = len(normal_data)

        train_end = int(
            total_normal * 0.40
        )

        valid_end = int(
            total_normal * 0.50
        )

        train_normal_data = normal_data.iloc[
            :train_end
        ].copy()

        valid_normal_data = normal_data.iloc[
            train_end:valid_end
        ].copy()

        # ----------------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------------

        bootstrap_size = max(
            1,
            int(
                len(train_normal_data)
                * BOOTSTRAP_FRACTION
            )
        )

        bootstrap_data = train_normal_data.iloc[
            :bootstrap_size
        ].copy()

        # Local train data = remaining 90%
        local_train_data = train_normal_data.iloc[
            bootstrap_size:
        ].copy()

        if len(local_train_data) == 0:

            local_train_data = train_normal_data.copy()

        client_data.append(
            {
                "client_id": client_id,
                "train": local_train_data,
                "valid": valid_normal_data,
                "bootstrap": bootstrap_data
            }
        )

    return client_data


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    set_seed(DATA_SEED)

    logger.info("Using device: %s", DEVICE)

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    client_data = load_client_data()

    # --------------------------------------------------------------
    # Create server bootstrap dataset
    # --------------------------------------------------------------

    bootstrap_parts = [
        item["bootstrap"]
        for item in client_data
    ]

    bootstrap_data = pd.concat(
        bootstrap_parts,
        ignore_index=True
    )

    # --------------------------------------------------------------
    # Same preprocessing as main.py
    #
    # 115 raw features
    #       ↓
    # log transform
    #       ↓
    # feature selection
    #       ↓
    # standard scaling
    #       ↓
    # 64 features
    # --------------------------------------------------------------

    data_processor = IoTDataProcessor(
        scaler="standard",
        use_log_transform=True,
        n_selected_features=TARGET_NUM_FEATURES
    )

    # Fit ONLY on bootstrap/server data
    data_processor.fit(bootstrap_data)

    # --------------------------------------------------------------
    # Prepare processed datasets
    # --------------------------------------------------------------

    processed_clients = []

    for item in client_data:

        train_processed = data_processor.transform(
            item["train"]
        )

        valid_processed = data_processor.transform(
            item["valid"]
        )

        processed_clients.append(
            {
                "client_id": item["client_id"],
                "train": train_processed,
                "valid": valid_processed
            }
        )

    # --------------------------------------------------------------
    # Select one client for overfitting test
    #
    # We test one client first so the graph clearly shows
    # local VAE training behaviour.
    # --------------------------------------------------------------

    client = processed_clients[0]

    logger.info(
        "Testing client: %s",
        client["client_id"]
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    train_dataset = IoTDataset(
        client["train"]
    )

    valid_dataset = IoTDataset(
        client["valid"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # --------------------------------------------------------------
    # Determine input dimension
    # --------------------------------------------------------------

    sample = client["train"]

    if isinstance(sample, pd.DataFrame):
        actual_dim_features = sample.shape[1]
    else:
        actual_dim_features = sample.shape[-1]

    logger.info(
        "Input dimension: %d",
        actual_dim_features
    )

    # --------------------------------------------------------------
    # Create VAE
    # --------------------------------------------------------------

    model = Autoencoder(
        input_dim=actual_dim_features,
        hidden_neus=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=actual_dim_features,
        use_sigmoid=False
    ).to(DEVICE)

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------------
    # History
    # --------------------------------------------------------------

    train_reconstruction_history = []
    train_kl_history = []
    train_total_history = []

    validation_reconstruction_history = []
    validation_kl_history = []
    validation_total_history = []

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    logger.info(
        "Starting overfitting test for %d epochs...",
        EPOCHS
    )

    for epoch in range(1, EPOCHS + 1):

        (
            train_reconstruction,
            train_kl,
            train_total
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        (
            validation_reconstruction,
            validation_kl,
            validation_total
        ) = evaluate_validation(
            model,
            valid_loader
        )

        # Save history
        train_reconstruction_history.append(
            train_reconstruction
        )

        train_kl_history.append(
            train_kl
        )

        train_total_history.append(
            train_total
        )

        validation_reconstruction_history.append(
            validation_reconstruction
        )

        validation_kl_history.append(
            validation_kl
        )

        validation_total_history.append(
            validation_total
        )

        logger.info(
            "Epoch %03d | "
            "Train Recon: %.6f | "
            "Val Recon: %.6f | "
            "Train KL: %.6f | "
            "Val KL: %.6f",
            epoch,
            train_reconstruction,
            validation_reconstruction,
            train_kl,
            validation_kl
        )

    # --------------------------------------------------------------
    # Save history
    # --------------------------------------------------------------

    history = pd.DataFrame(
        {
            "epoch": range(1, EPOCHS + 1),

            "train_reconstruction_mse":
                train_reconstruction_history,

            "validation_reconstruction_mse":
                validation_reconstruction_history,

            "train_kl":
                train_kl_history,

            "validation_kl":
                validation_kl_history,

            "train_total_loss":
                train_total_history,

            "validation_total_loss":
                validation_total_history
        }
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        "overfitting_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    logger.info(
        "History saved to: %s",
        history_path
    )

    # --------------------------------------------------------------
    # Plot reconstruction MSE
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_reconstruction_mse"],
        label="Train Reconstruction MSE"
    )

    plt.plot(
        history["epoch"],
        history["validation_reconstruction_mse"],
        label="Validation Reconstruction MSE"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Reconstruction MSE")

    plt.title(
        "VAE Overfitting Test: Train vs Validation Reconstruction MSE"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "overfitting_train_vs_validation.png"
    )

    plt.savefig(
        plot_path,
        dpi=300
    )

    plt.show()

    logger.info(
        "Plot saved to: %s",
        plot_path
    )

    # --------------------------------------------------------------
    # KL plot
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_kl"],
        label="Train KL"
    )

    plt.plot(
        history["epoch"],
        history["validation_kl"],
        label="Validation KL"
    )

    plt.xlabel("Epoch")

    plt.ylabel("KL Divergence")

    plt.title(
        "VAE KL Loss During Training"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    kl_plot_path = os.path.join(
        OUTPUT_DIR,
        "overfitting_kl_loss.png"
    )

    plt.savefig(
        kl_plot_path,
        dpi=300
    )

    plt.show()

    logger.info(
        "KL plot saved to: %s",
        kl_plot_path
    )

    # --------------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------------

    min_val_epoch = (
        history[
            "validation_reconstruction_mse"
        ].idxmin() + 1
    )

    min_val_mse = (
        history[
            "validation_reconstruction_mse"
        ].min()
    )

    final_train_mse = (
        history[
            "train_reconstruction_mse"
        ].iloc[-1]
    )

    final_val_mse = (
        history[
            "validation_reconstruction_mse"
        ].iloc[-1]
    )

    logger.info(
        "Best validation reconstruction MSE: %.6f "
        "at epoch %d",
        min_val_mse,
        min_val_epoch
    )

    logger.info(
        "Final Train Reconstruction MSE: %.6f",
        final_train_mse
    )

    logger.info(
        "Final Validation Reconstruction MSE: %.6f",
        final_val_mse
    )

    if (
        final_train_mse < min_val_mse
        and final_val_mse > min_val_mse
    ):

        logger.warning(
            "Possible overfitting detected: "
            "validation MSE increased after reaching its minimum."
        )

    else:

        logger.info(
            "No clear overfitting pattern detected "
            "from the final epoch comparison."
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
