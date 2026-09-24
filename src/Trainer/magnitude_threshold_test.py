"""
Offline Magnitude Threshold Sensitivity Test

Purpose:
    Find the lowest magnitude attack factor at which
    SecurityBuffer stops flagging the magnitude condition.

Test:
    Magnitude factors:
        1.0x
        1.5x
        2.0x
        3.0x
        5.0x

Only magnitude is varied.

SecurityBuffer:
    min_history = 2

The test uses:
    - Saved global VAE checkpoint
    - Real Client-5 local update
    - Two clean historical magnitude observations
    - Current SecurityBuffer magnitude evaluation
"""

import os
import sys
import copy
import json
import random
import logging

# Add project src directory to Python import path
sys.path.insert(0, "/content/fedmse/src")

import numpy as np
import torch

from torch.utils.data import DataLoader

from DataLoader.dataloader import (
    load_data,
    IoTDataset,
    IoTDataProcessor
)

from Model import Autoencoder

from Trainer import ClientTrainer

from Trainer.security_buffer import SecurityBuffer

from Evaluator.comet_logger import (
    create_magnitude_experiment,
    log_magnitude_threshold_metrics,
    finish_experiment
)


# ================================================================
# CONFIGURATION
# ================================================================

CONFIG_FILE = (
    "/content/fedmse/Configuration/"
    "scen2-nba-iot-10clients.json"
)

CHECKPOINT_PATH = (
    "/content/fedmse/results/"
    "autoencoder_run1_checkpoint.pth"
)

CLIENT_ID = "Client-5"

BATCH_SIZE = 64

LR_RATE = 1e-5

EPOCHS = 15

LATENT_DIM = 16

TARGET_NUM_FEATURES = 64

VAE_KL_WEIGHT = 0.0001

DATA_SEED = 1234

# Exactly two clean history observations
MIN_HISTORY = 2

# Magnitude factors to test
MAGNITUDE_FACTORS = [
    1.0,
    0.5,
    0.2,
    3.0,
    5.0
]


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ================================================================
# SEED
# ================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ================================================================
# LOAD CLIENT DATA
# ================================================================

def prepare_client():

    with open(
        CONFIG_FILE,
        "r"
    ) as f:

        config = json.load(f)

    client_config = None

    for client in config["devices_list"]:

        if client["name"] == CLIENT_ID:

            client_config = client

            break

    if client_config is None:

        raise ValueError(
            f"{CLIENT_ID} not found in configuration."
        )

    data_path = config["data_path"]

    normal_path = os.path.join(
        data_path,
        client_config["normal_data_path"]
    )

    normal_data = (
        load_data(normal_path)
        .sample(
            frac=1,
            random_state=DATA_SEED
        )
        .reset_index(drop=True)
    )

    train_normal_size = int(
        0.4 * len(normal_data)
    )

    valid_normal_size = int(
        0.1 * len(normal_data)
    )

    train_normal_data = (
        normal_data[
            :train_normal_size
        ]
        .reset_index(drop=True)
    )

    valid_normal_data = (
        normal_data[
            train_normal_size:
            train_normal_size + valid_normal_size
        ]
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Bootstrap data
    # ------------------------------------------------------------

    bootstrap_fraction = 0.10

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
        .reset_index(drop=True)
    )

    local_train_data = (
        train_normal_data[
            bootstrap_size:
        ]
        .reset_index(drop=True)
    )

    if len(local_train_data) == 0:

        local_train_data = train_normal_data

    # ------------------------------------------------------------
    # Fit same preprocessing pipeline as main.py
    # ------------------------------------------------------------

    processor = IoTDataProcessor(
        scaler="standard",
        use_log_transform=True,
        n_selected_features=TARGET_NUM_FEATURES
    )

    # Fit on bootstrap data
    processor.fit_transform(
        bootstrap_data
    )

    processed_train, train_labels = (
        processor.transform(
            local_train_data,
            type="normal"
        )
    )

    processed_valid, valid_labels = (
        processor.transform(
            valid_normal_data,
            type="normal"
        )
    )

    train_dataset = IoTDataset(
        processed_train,
        train_labels
    )

    valid_dataset = IoTDataset(
        processed_valid,
        valid_labels
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
        shuffle=False
    )

    return (
        train_loader,
        valid_loader,
        processed_train.shape[1]
    )


# ================================================================
# BUILD MODEL
# ================================================================

def build_model(
    input_dim,
    device
):

    model = Autoencoder(
        input_dim=input_dim,
        hidden_neus=32,
        latent_dim=LATENT_DIM,
        output_dim=input_dim,
        use_sigmoid=False
    )

    model.to(device)

    return model


# ================================================================
# GENERATE REAL CLIENT UPDATE
# ================================================================

def generate_client_update(
    global_model,
    train_loader,
    valid_loader,
    device
):

    local_model = copy.deepcopy(
        global_model
    )

    trainer = ClientTrainer(
        model=local_model,
        client_id=CLIENT_ID,
        train_loader=train_loader,
        epoch=EPOCHS,
        lr_rate=LR_RATE,
        update_type="fedopt",
        device=str(device),
        save_dir="/tmp/magnitude_threshold_test",
        kl_weight=VAE_KL_WEIGHT
    )

    trainer.run(
        train_loader,
        valid_loader
    )

    update = {

        "client_id":
            CLIENT_ID,

        "weights":
            copy.deepcopy(
                trainer.get_parameters()
            ),

        "train_time":
            trainer.train_time,

        "dataset_size":
            trainer.dataset_size,

        "val_mse_list":
            copy.deepcopy(
                trainer.val_mse_list
            ),

        "val_loss":
            trainer.val_loss
    }

    return update


# ================================================================
# MANIPULATE MAGNITUDE
# ================================================================

def manipulate_magnitude(
    update,
    factor
):

    manipulated = copy.deepcopy(
        update
    )

    for name, tensor in (
        manipulated["weights"].items()
    ):

        if torch.is_tensor(tensor):

            manipulated["weights"][name] = (
                tensor * factor
            )

    return manipulated


# ================================================================
# MAIN TEST
# ================================================================

def main():

    set_seed(DATA_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(
        f"Using device: {device}"
    )

    # ------------------------------------------------------------
    # Load client data
    # ------------------------------------------------------------

    (
        train_loader,
        valid_loader,
        input_dim
    ) = prepare_client()

    logging.info(
        f"Client: {CLIENT_ID}"
    )

    logging.info(
        f"Input dimension: {input_dim}"
    )

    # ------------------------------------------------------------
    # Load global checkpoint
    # ------------------------------------------------------------

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    global_model = build_model(
        input_dim,
        device
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    global_model.load_state_dict(
        checkpoint
    )

    global_model.eval()

    logging.info(
        "Global checkpoint loaded successfully."
    )

    # ------------------------------------------------------------
    # Generate one real clean client update
    # ------------------------------------------------------------

    clean_update = generate_client_update(
        global_model,
        train_loader,
        valid_loader,
        device
    )

    # ------------------------------------------------------------
    # Create SecurityBuffer
    # ------------------------------------------------------------

    security_buffer = SecurityBuffer(
        global_model=global_model,
        window_size=5,
        latency_threshold=1.5,
        alpha=0.2,
        beta=0.01
    )

    # ------------------------------------------------------------
    # Get clean magnitude
    # ------------------------------------------------------------

    clean_magnitude = (
        security_buffer.compute_update_magnitude(
            clean_update["weights"]
        )
    )

    logging.info(
        f"Clean update magnitude: "
        f"{clean_magnitude:.6f}"
    )

    # ------------------------------------------------------------
    # Create TWO clean history observations
    #
    # SecurityBuffer requires min_history = 2
    # before Magnitude_Fail can be evaluated.
    # ------------------------------------------------------------

    client_history = (
        security_buffer.client_history[
            CLIENT_ID
        ]
    )

    client_history["magnitude"] = [

        float(clean_magnitude),

        float(clean_magnitude)
    ]

    logging.info(
        "Inserted 2 clean magnitude history observations."
    )

    # ------------------------------------------------------------
    # Comet
    # ------------------------------------------------------------

    experiment = create_magnitude_experiment(
        client_id=CLIENT_ID,
        checkpoint_path=CHECKPOINT_PATH,
        magnitude_factors=MAGNITUDE_FACTORS
    )

    # ------------------------------------------------------------
    # Factor sweep
    # ------------------------------------------------------------

    results = []

    for factor in MAGNITUDE_FACTORS:

        attacked_update = (
            manipulate_magnitude(
                clean_update,
                factor
            )
        )

        current_magnitude = (
            security_buffer.compute_update_magnitude(
                attacked_update["weights"]
            )
        )

        # Use current SecurityBuffer threshold logic
        history_values = (
            client_history["magnitude"]
        )

        magnitude_threshold = (
            security_buffer._adaptive_threshold(
                history_values,
                security_buffer.magnitude_threshold
            )
        )

        magnitude_fail = (
            current_magnitude
            >
            magnitude_threshold
        )

        detection_rate = (
            100.0
            if magnitude_fail
            else 0.0
        )

        result = {

            "attack_factor":
                factor,

            "magnitude":
                current_magnitude,

            "magnitude_threshold":
                magnitude_threshold,

            "magnitude_fail":
                int(magnitude_fail),

            "magnitude_detection_rate":
                detection_rate
        }

        results.append(
            result
        )

        log_magnitude_threshold_metrics(
            experiment=experiment,
            attack_factor=factor,
            magnitude=current_magnitude,
            magnitude_threshold=magnitude_threshold,
            magnitude_fail=magnitude_fail
        )

        logging.info(
            f"Factor={factor:.1f}x | "
            f"Magnitude={current_magnitude:.6f} | "
            f"Threshold={magnitude_threshold:.6f} | "
            f"Magnitude_Fail={magnitude_fail}"
        )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    output_path = (
        "/content/fedmse/"
        "magnitude_threshold_results.csv"
    )

    import pandas as pd

    pd.DataFrame(
        results
    ).to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"Results saved to:\n"
        f"{output_path}"
    )

    # ------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------

    passing_factors = [
        r["attack_factor"]
        for r in results
        if r["magnitude_fail"] == 0
    ]

    if passing_factors:

        lowest_undetected = min(
            passing_factors
        )

        logging.info(
            "===================================================="
        )

        logging.info(
            f"Lowest tested factor with "
            f"Magnitude_Fail=False: "
            f"{lowest_undetected:.1f}x"
        )

        logging.info(
            "===================================================="
        )

    else:

        logging.info(
            "All tested factors were detected "
            "by the magnitude condition."
        )

    finish_experiment(
        experiment
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()
