"""
VAE Overfitting Test

Uses the SAME:
    - data loading
    - client selection
    - train/validation split
    - bootstrap dataset
    - preprocessing
    - VAE architecture
    - batch size
    - learning rate
    - KL weight

from main.py.

No:
    - FedOpt
    - Security Buffer
    - malicious update
    - federated rounds
    - matplotlib

Results are logged to Comet ML epoch-by-epoch.
"""

import os
import json
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from DataLoader.dataloader import (
    load_data,
    IoTDataset,
    IoTDataProcessor
)

from Model import (
    Autoencoder
)

from Evaluator.comet_logger import (
    create_experiment,
    finish_experiment
)


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ================================================================
# SAME HYPERPARAMETERS AS main.py
# ================================================================

epoch = 15

lr_rate = 1e-5

shrink_dim = 16

network_size = 10

data_seed = 1234

batch_size = 64

target_num_features = 64

bootstrap_fraction = 0.10

vae_kl_weight = 0.0001

initial_epochs = 1

threshold_val = 0.2


config_file = (
    "/content/fedmse/Configuration/"
    "scen2-nba-iot-10clients.json"
)


# ================================================================
# OVERFITTING TEST SETTINGS
# ================================================================

# Use more epochs than the normal FL experiment so that
# overfitting can become visible if it exists.
OVERFITTING_EPOCHS = 50

# Client to test.
# Change only this value if you want another client.
TEST_CLIENT_INDEX = 0


# ================================================================
# RANDOM SEED
# ================================================================

def set_seeds(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ================================================================
# VAE KL LOSS
# ================================================================

def compute_kl_loss(
    mu,
    logvar
):

    logvar = torch.clamp(
        logvar,
        min=-10.0,
        max=10.0
    )

    kl_loss = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl_loss.sum(
        dim=1
    ).mean()


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device
):

    model.train()

    total_reconstruction_loss = 0.0

    total_kl_loss = 0.0

    total_samples = 0

    for batch in train_loader:

        inputs = (
            batch[0].to(device)
            if isinstance(
                batch,
                (list, tuple)
            )
            else batch.to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            inputs
        )

        # ---------------------------------------------------------
        # VAE output:
        #
        # reconstruction, mu, logvar
        # ---------------------------------------------------------

        reconstruction = outputs[0]

        mu = outputs[1]

        logvar = outputs[2]

        reconstruction_loss = F.mse_loss(
            reconstruction,
            inputs,
            reduction="mean"
        )

        kl_loss = compute_kl_loss(
            mu,
            logvar
        )

        total_loss = (
            reconstruction_loss
            +
            vae_kl_weight
            *
            kl_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        current_batch_size = (
            inputs.size(0)
        )

        total_reconstruction_loss += (
            reconstruction_loss.item()
            *
            current_batch_size
        )

        total_kl_loss += (
            kl_loss.item()
            *
            current_batch_size
        )

        total_samples += (
            current_batch_size
        )

    mean_reconstruction_loss = (
        total_reconstruction_loss
        /
        max(
            total_samples,
            1
        )
    )

    mean_kl_loss = (
        total_kl_loss
        /
        max(
            total_samples,
            1
        )
    )

    mean_total_loss = (
        mean_reconstruction_loss
        +
        vae_kl_weight
        *
        mean_kl_loss
    )

    return (
        mean_reconstruction_loss,
        mean_kl_loss,
        mean_total_loss
    )


# ================================================================
# VALIDATION
# ================================================================

def evaluate_validation(
    model,
    valid_loader,
    device
):

    model.eval()

    total_reconstruction_loss = 0.0

    total_kl_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for batch in valid_loader:

            inputs = (
                batch[0].to(device)
                if isinstance(
                    batch,
                    (list, tuple)
                )
                else batch.to(device)
            )

            outputs = model(
                inputs
            )

            reconstruction = outputs[0]

            mu = outputs[1]

            logvar = outputs[2]

            reconstruction_loss = F.mse_loss(
                reconstruction,
                inputs,
                reduction="mean"
            )

            kl_loss = compute_kl_loss(
                mu,
                logvar
            )

            current_batch_size = (
                inputs.size(0)
            )

            total_reconstruction_loss += (
                reconstruction_loss.item()
                *
                current_batch_size
            )

            total_kl_loss += (
                kl_loss.item()
                *
                current_batch_size
            )

            total_samples += (
                current_batch_size
            )

    mean_reconstruction_loss = (
        total_reconstruction_loss
        /
        max(
            total_samples,
            1
        )
    )

    mean_kl_loss = (
        total_kl_loss
        /
        max(
            total_samples,
            1
        )
    )

    mean_total_loss = (
        mean_reconstruction_loss
        +
        vae_kl_weight
        *
        mean_kl_loss
    )

    return (
        mean_reconstruction_loss,
        mean_kl_loss,
        mean_total_loss
    )


# ================================================================
# MAIN
# ================================================================

def main():

    set_seeds(
        data_seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(
        f"Execution started on target device: "
        f"{device}"
    )

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    with open(
        config_file,
        "r"
    ) as config_f:

        config = json.load(
            config_f
        )

    # ============================================================
    # SAME CLIENT SELECTION AS main.py
    # ============================================================

    devices_list = random.sample(
        config["devices_list"],
        network_size
    )

    # ============================================================
    # STEP 1:
    # LOAD RAW CLIENT DATA
    # ============================================================

    raw_client_data = []

    logging.info(
        "Loading client datasets..."
    )

    for dev in devices_list:

        normal_data_path = os.path.join(
            config["data_path"],
            dev["normal_data_path"]
        )

        normal_data = (
            load_data(
                normal_data_path
            )
            .sample(
                frac=1,
                random_state=data_seed
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Same normal split as main.py
        # --------------------------------------------------------

        train_normal_size = int(
            0.4
            *
            len(normal_data)
        )

        valid_normal_size = int(
            0.1
            *
            len(normal_data)
        )

        train_normal_data = (
            normal_data[
                :train_normal_size
            ]
            .reset_index(
                drop=True
            )
        )

        valid_normal_data = (
            normal_data[
                train_normal_size:
                train_normal_size
                +
                valid_normal_size
            ]
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Same bootstrap logic as main.py
        # --------------------------------------------------------

        bootstrap_size = max(
            1,
            int(
                bootstrap_fraction
                *
                len(train_normal_data)
            )
        )

        bootstrap_data = (
            train_normal_data[
                :bootstrap_size
            ]
            .reset_index(
                drop=True
            )
        )

        local_train_data = (
            train_normal_data[
                bootstrap_size:
            ]
            .reset_index(
                drop=True
            )
        )

        if len(local_train_data) == 0:

            local_train_data = (
                train_normal_data
            )

        raw_client_data.append({

            "device":
                dev["name"],

            "bootstrap_data":
                bootstrap_data,

            "train_data":
                local_train_data,

            "valid_data":
                valid_normal_data
        })

    # ============================================================
    # STEP 2:
    # SERVER BOOTSTRAP DATASET
    # ============================================================

    bootstrap_server_dataframe = (
        __import__("pandas")
        .concat(
            [
                client["bootstrap_data"]
                for client in raw_client_data
            ],
            ignore_index=True
        )
    )

    logging.info(
        f"Initial server bootstrap dataset created | "
        f"Samples: "
        f"{len(bootstrap_server_dataframe)}"
    )

    # ============================================================
    # STEP 3:
    # SAME PREPROCESSING AS main.py
    # ============================================================

    data_processor = IoTDataProcessor(
        scaler="standard",
        use_log_transform=True,
        n_selected_features=target_num_features
    )

    (
        processed_bootstrap_data,
        bootstrap_label
    ) = data_processor.fit_transform(
        bootstrap_server_dataframe
    )

    actual_dim_features = (
        processed_bootstrap_data.shape[1]
    )

    logging.info(
        f"Feature pipeline complete | "
        f"Original features: "
        f"{bootstrap_server_dataframe.shape[1]} | "
        f"Selected features: "
        f"{actual_dim_features}"
    )

    # ============================================================
    # STEP 4:
    # PROCESS EVERY CLIENT
    # ============================================================

    client_info = []

    for client in raw_client_data:

        (
            processed_train_data,
            train_label
        ) = data_processor.transform(
            client["train_data"],
            type="normal"
        )

        (
            processed_valid_data,
            valid_label
        ) = data_processor.transform(
            client["valid_data"],
            type="normal"
        )

        train_dataset = IoTDataset(
            processed_train_data,
            train_label
        )

        valid_dataset = IoTDataset(
            processed_valid_data,
            valid_label
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False
        )

        client_info.append({

            "device":
                client["device"],

            "train_loader":
                train_loader,

            "valid_loader":
                valid_loader
        })

    # ============================================================
    # SELECT CLIENT FOR OVERFITTING TEST
    # ============================================================

    test_client = client_info[
        TEST_CLIENT_INDEX
    ]

    client_name = (
        test_client["device"]
    )

    train_loader = (
        test_client["train_loader"]
    )

    valid_loader = (
        test_client["valid_loader"]
    )

    logging.info(
        f"===================================================="
    )

    logging.info(
        f"OVERFITTING TEST CLIENT: "
        f"{client_name}"
    )

    logging.info(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    logging.info(
        f"Validation samples: "
        f"{len(valid_loader.dataset)}"
    )

    logging.info(
        f"===================================================="
    )

    # ============================================================
    # CREATE VAE
    # ============================================================

    model = Autoencoder(
        input_dim=actual_dim_features,
        hidden_neus=32,
        latent_dim=shrink_dim,
        output_dim=actual_dim_features,
        use_sigmoid=False
    )

    model.to(
        device
    )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_rate
    )

    # ============================================================
    # COMET EXPERIMENT
    # ============================================================

    experiment = create_experiment(
        model_type="vae_overfitting_test",
        run_number=1,
        run_seed=data_seed,
        num_rounds=OVERFITTING_EPOCHS,
        epoch=OVERFITTING_EPOCHS,
        learning_rate=lr_rate,
        shrink_dim=shrink_dim,
        batch_size=batch_size,
        latency_threshold=1.5,
        server_lr=0.01,
        update_type="local_vae",
        network_size=1,
        raw_features=actual_dim_features,
        timing_attack_client="none",
        timing_attack_start_round=0
    )

    # Additional experiment parameters
    experiment.log_parameters({

        "test_type":
            "VAE overfitting",

        "client":
            client_name,

        "train_samples":
            len(train_loader.dataset),

        "validation_samples":
            len(valid_loader.dataset),

        "bootstrap_fraction":
            bootstrap_fraction,

        "target_num_features":
            target_num_features,

        "use_log_transform":
            True,

        "scaler":
            "standard",

        "feature_selection":
            "Unsupervised variance-based selection",

        "vae_kl_weight":
            vae_kl_weight
    })

    # ============================================================
    # TRAINING HISTORY
    # ============================================================

    best_validation_mse = float(
        "inf"
    )

    best_epoch = 0

    # ============================================================
    # OVERFITTING TRAINING
    # ============================================================

    try:

        for current_epoch in range(
            1,
            OVERFITTING_EPOCHS + 1
        ):

            (
                train_reconstruction_mse,
                train_kl,
                train_total_loss
            ) = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device
            )

            (
                validation_reconstruction_mse,
                validation_kl,
                validation_total_loss
            ) = evaluate_validation(
                model=model,
                valid_loader=valid_loader,
                device=device
            )

            # ----------------------------------------------------
            # Track best validation epoch
            # ----------------------------------------------------

            if (
                validation_reconstruction_mse
                <
                best_validation_mse
            ):

                best_validation_mse = (
                    validation_reconstruction_mse
                )

                best_epoch = (
                    current_epoch
                )

            # ----------------------------------------------------
            # Comet epoch-wise logging
            # ----------------------------------------------------

            experiment.log_metrics(

                {

                    "train_reconstruction_mse":
                        float(
                            train_reconstruction_mse
                        ),

                    "validation_reconstruction_mse":
                        float(
                            validation_reconstruction_mse
                        ),

                    "train_kl":
                        float(
                            train_kl
                        ),

                    "validation_kl":
                        float(
                            validation_kl
                        ),

                    "train_total_loss":
                        float(
                            train_total_loss
                        ),

                    "validation_total_loss":
                        float(
                            validation_total_loss
                        ),

                    "train_validation_gap":
                        float(
                            validation_reconstruction_mse
                            -
                            train_reconstruction_mse
                        )
                },

                step=current_epoch
            )

            logging.info(

                f"Epoch "
                f"{current_epoch:03d}/{OVERFITTING_EPOCHS} | "

                f"Train Recon MSE: "
                f"{train_reconstruction_mse:.8f} | "

                f"Val Recon MSE: "
                f"{validation_reconstruction_mse:.8f} | "

                f"Train KL: "
                f"{train_kl:.8f} | "

                f"Val KL: "
                f"{validation_kl:.8f}"
            )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        experiment.log_parameters({

            "best_validation_epoch":
                best_epoch,

            "best_validation_reconstruction_mse":
                float(
                    best_validation_mse
                )
        })

        logging.info(
            "===================================================="
        )

        logging.info(
            f"Best validation reconstruction MSE: "
            f"{best_validation_mse:.8f}"
        )

        logging.info(
            f"Best validation epoch: "
            f"{best_epoch}"
        )

        logging.info(
            "===================================================="
        )

    finally:

        finish_experiment(
            experiment
        )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()
