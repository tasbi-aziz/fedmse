"""
Main Execution Script for Federated Learning with FedOpt,
VAE-based Anomaly Detection, and Security Buffer Routing.

Pipeline:

    Client normal data
        ↓
    Bootstrap samples from clients
        ↓
    Server-side initial dataset
        ↓
    Common preprocessing
        ↓
    Log transformation
        ↓
    Unsupervised feature selection
        ↓
    Scaling
        ↓
    Initial global model training
        ↓
    Local VAE training
        ↓
    Client update:
        - weights
        - dataset size
        - training time
        - validation loss
        - 5-fold validation MSE list
        ↓
    Security Buffer
        ↓
    Direct / Secondary / Quarantine
        ↓
    Server validation for secondary updates
        ↓
    Adaptive weight factor
        ↓
    FedOpt aggregation
        ↓
    Global evaluation
        ↓
    Client-wise evaluation using CURRENT GLOBAL MODEL
        ↓
    Repeat
"""

import os
import json
import argparse
import copy
import random
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from torch.utils.data import (
    DataLoader,
    ConcatDataset
)

from DataLoader.dataloader import (
    load_data,
    IoTDataset,
    IoTDataProcessor
)

from Trainer import (
    ClientTrainer,
    GlobalAggregator
)

from Trainer.security_buffer import (
    SecurityBuffer
)

from Model import (
    Shrink_Autoencoder,
    Autoencoder
)


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ================================================================
# GLOBAL HYPERPARAMETERS
# ================================================================

num_participants = 1.0

epoch = 15

num_rounds = 15

lr_rate = 1e-5

# VAE latent dimension
shrink_dim = 16

threshold_val = 0.2

network_size = 10

data_seed = 1234

num_runs = 5

batch_size = 64

# Number of features after unsupervised feature selection
target_num_features = 64

# ------------------------------------------------
# Fraction of each client's normal training data
# used to create the initial server dataset.
# ------------------------------------------------

bootstrap_fraction = 0.10

# ------------------------------------------------
# Initial server training epochs
# ------------------------------------------------

initial_epochs = 1

# ------------------------------------------------
# VAE KL weight
# ------------------------------------------------

vae_kl_weight = 0.0001

# ------------------------------------------------
# Server-side validation weighting
# ------------------------------------------------

minimum_weight_factor = 0.10

quarantine_weight_factor = 0.05

validation_rejection_ratio = 4.0


config_file = (
    "/content/fedmse/Configuration/"
    "scen2-nba-iot-10clients.json"
)


# ================================================================
# RANDOM SEED
# ================================================================

def set_seeds(seed):
    """
    Sets deterministic random seeds across all libraries.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ================================================================
# MODEL OUTPUT EXTRACTION
# ================================================================

def extract_reconstructed_output(outputs):
    """
    Extracts reconstruction from model output.

    For VAE:
        (reconstruction, mu, logvar)

    Therefore reconstruction = outputs[0].
    """

    if isinstance(
        outputs,
        (tuple, list)
    ):

        if len(outputs) == 0:

            raise ValueError(
                "Model returned an empty tuple/list."
            )

        return outputs[0]

    return outputs


# ================================================================
# GLOBAL MSE
# ================================================================

def evaluate_global_mse(
    global_model,
    val_loader,
    device="cpu"
):
    """
    Computes reconstruction MSE of the global model
    on the server validation dataset.
    """

    if (
        global_model is None
        or val_loader is None
    ):

        return 0.0

    global_model.eval()

    total_loss = 0.0

    total_samples = 0

    criterion = nn.MSELoss()

    with torch.no_grad():

        for batch in val_loader:

            inputs = (
                batch[0].to(device)
                if isinstance(
                    batch,
                    (list, tuple)
                )
                else batch.to(device)
            )

            outputs = global_model(
                inputs
            )

            reconstructed = (
                extract_reconstructed_output(
                    outputs
                )
            )

            loss = criterion(
                reconstructed,
                inputs
            )

            total_loss += (
                loss.item()
                *
                inputs.size(0)
            )

            total_samples += (
                inputs.size(0)
            )

    return (
        total_loss
        /
        max(
            total_samples,
            1
        )
    )


# ================================================================
# SAMPLE-WISE RECONSTRUCTION ERRORS
# ================================================================

def compute_reconstruction_errors(
    model,
    data_loader,
    device="cpu"
):
    """
    Calculates sample-wise MSE reconstruction errors.
    """

    model.eval()

    errors = []

    criterion = nn.MSELoss(
        reduction='none'
    )

    with torch.no_grad():

        for batch in data_loader:

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

            reconstructed = (
                extract_reconstructed_output(
                    outputs
                )
            )

            loss = criterion(
                reconstructed,
                inputs
            ).mean(
                dim=1
            )

            errors.extend(
                loss.cpu().numpy()
            )

    return np.array(
        errors
    )


# ================================================================
# ANOMALY DETECTION EVALUATION
# ================================================================

def evaluate_anomaly_detection(
    model,
    test_loader,
    device="cpu"
):
    """
    Evaluates anomaly detection performance.

    Returns:
        Precision
        Recall
        F1
        ROC-AUC
        Threshold
        Confusion Matrix
    """

    model.eval()

    y_true = []

    reconstruction_errors = []

    criterion = nn.MSELoss(
        reduction='none'
    )

    with torch.no_grad():

        for batch in test_loader:

            inputs = batch[0].to(
                device
            )

            labels = (
                batch[1]
                .cpu()
                .numpy()
            )

            outputs = model(
                inputs
            )

            reconstructed = (
                extract_reconstructed_output(
                    outputs
                )
            )

            loss = criterion(
                reconstructed,
                inputs
            ).mean(
                dim=1
            ).cpu().numpy()

            reconstruction_errors.extend(
                loss
            )

            y_true.extend(
                labels
            )

    y_true = np.array(
        y_true
    )

    reconstruction_errors = np.array(
        reconstruction_errors
    )

    # -------------------------------------------------------------
    # Dynamic threshold from normal samples
    # -------------------------------------------------------------

    normal_errors = (
        reconstruction_errors[
            y_true == 0
        ]
    )

    if len(normal_errors) > 0:

        threshold = np.percentile(
            normal_errors,
            95
        )

    else:

        threshold = (
            np.mean(
                reconstruction_errors
            )
            +
            np.std(
                reconstruction_errors
            )
        )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    y_pred = (
        reconstruction_errors
        >
        threshold
    ).astype(
        int
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_true,
            reconstruction_errors
        )

    except ValueError:

        auc = 0.5

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    metrics = {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "auc_roc":
            float(auc),

        "threshold":
            float(threshold),

        "tp":
            int(tp),

        "fp":
            int(fp),

        "tn":
            int(tn),

        "fn":
            int(fn)
    }

    return metrics


# ================================================================
# CLIENT-WISE GLOBAL MODEL EVALUATION
# ================================================================

def evaluate_clientwise_auc(
    global_model,
    client_info,
    device="cpu"
):
    """
    Evaluates the CURRENT GLOBAL MODEL separately on every
    client's own test dataset.

    IMPORTANT:

    The model being evaluated is the GLOBAL MODEL after
    the current round's FedOpt aggregation.

    Each client uses:

        test_normal.csv
              +
        abnormal.csv

    Therefore each client gets its own AUC.

    Returns:

        {
            "Client-1": {
                "auc": ...,
                "precision": ...,
                "recall": ...,
                "f1": ...
            },
            ...
        }
    """

    client_results = {}

    logging.info(
        "===================================================="
    )

    logging.info(
        "[Client-wise Evaluation] "
        "Evaluating CURRENT GLOBAL MODEL on each client"
    )

    logging.info(
        "===================================================="
    )

    # -------------------------------------------------------------
    # Freeze current global weights
    # -------------------------------------------------------------

    global_state = copy.deepcopy(
        global_model.state_dict()
    )

    # -------------------------------------------------------------
    # Evaluate each client's own test dataset
    # -------------------------------------------------------------

    for client in client_info:

        client_id = client["device"]

        test_loader = client["test_loader"]

        # ---------------------------------------------------------
        # Create a copy of the GLOBAL MODEL.
        #
        # This represents sending the current global weights
        # to this client for evaluation.
        # ---------------------------------------------------------

        client_eval_model = copy.deepcopy(
            global_model
        ).to(device)

        client_eval_model.load_state_dict(
            global_state
        )

        client_eval_model.eval()

        # ---------------------------------------------------------
        # Evaluate
        # ---------------------------------------------------------

        metrics = evaluate_anomaly_detection(
            client_eval_model,
            test_loader,
            device=device
        )

        client_results[
            client_id
        ] = {

            "auc":
                float(
                    metrics["auc_roc"]
                ),

            "precision":
                float(
                    metrics["precision"]
                ),

            "recall":
                float(
                    metrics["recall"]
                ),

            "f1":
                float(
                    metrics["f1_score"]
                )
        }

        logging.info(
            f"[Client-wise AUC] "
            f"{client_id} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        # ---------------------------------------------------------
        # Release temporary evaluation model
        # ---------------------------------------------------------

        del client_eval_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Mean client AUC
    # -------------------------------------------------------------

    client_auc_values = [
        result["auc"]
        for result in client_results.values()
    ]

    mean_client_auc = (
        float(
            np.mean(
                client_auc_values
            )
        )
        if client_auc_values
        else 0.0
    )

    logging.info(
        f"[Client-wise Evaluation] "
        f"Mean Client AUC: "
        f"{mean_client_auc:.4f}"
    )

    logging.info(
        "===================================================="
    )

    return client_results


# ================================================================
# SERVER VALIDATION OF SECONDARY UPDATE
# ================================================================

def validate_secondary_update(
    update,
    global_model,
    server_val_loader,
    current_global_mse,
    device="cpu"
):
    """
    Runs a suspicious / delayed client model on the
    server validation dataset.
    """

    if (
        update is None
        or "weights" not in update
        or global_model is None
        or server_val_loader is None
    ):

        return (
            quarantine_weight_factor,
            float("inf"),
            "QUARANTINE"
        )

    candidate_model = copy.deepcopy(
        global_model
    ).to(device)

    try:

        candidate_model.load_state_dict(
            update["weights"]
        )

    except Exception as exc:

        logging.warning(
            f"[Server Validation] "
            f"Failed to load Client "
            f"{update.get('client_id', 'unknown')} "
            f"weights: {exc}"
        )

        return (
            quarantine_weight_factor,
            float("inf"),
            "QUARANTINE"
        )

    candidate_model.eval()

    criterion = nn.MSELoss()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for batch in server_val_loader:

            inputs = (
                batch[0].to(device)
                if isinstance(
                    batch,
                    (list, tuple)
                )
                else batch.to(device)
            )

            outputs = candidate_model(
                inputs
            )

            reconstructed = (
                extract_reconstructed_output(
                    outputs
                )
            )

            loss = criterion(
                reconstructed,
                inputs
            )

            total_loss += (
                loss.item()
                *
                inputs.size(0)
            )

            total_samples += (
                inputs.size(0)
            )

    candidate_mse = (
        total_loss
        /
        max(
            total_samples,
            1
        )
    )

    if not math.isfinite(
        candidate_mse
    ):

        return (
            quarantine_weight_factor,
            candidate_mse,
            "QUARANTINE"
        )

    if (
        not math.isfinite(
            current_global_mse
        )
        or current_global_mse <= 0
    ):

        return (
            minimum_weight_factor,
            candidate_mse,
            "SECONDARY_ACCEPTED"
        )

    performance_ratio = (
        current_global_mse
        /
        (
            candidate_mse
            +
            1e-8
        )
    )

    if (
        candidate_mse
        >
        current_global_mse
        *
        validation_rejection_ratio
    ):

        return (
            quarantine_weight_factor,
            candidate_mse,
            "QUARANTINE"
        )

    performance_ratio = max(
        performance_ratio,
        0.0
    )

    weight_factor = math.sqrt(
        performance_ratio
    )

    weight_factor = float(
        np.clip(
            weight_factor,
            minimum_weight_factor,
            1.0
        )
    )

    return (
        weight_factor,
        candidate_mse,
        "SECONDARY_ACCEPTED"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Federated Learning Evaluation Pipeline "
            "with VAE, FedOpt and Security Buffer"
        )
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # 1.5 sec = FAST/SLOW classification threshold
    # -------------------------------------------------------------

    parser.add_argument(
        "--latency_threshold",
        type=float,
        default=1.5,
        help=(
            "FAST/SLOW arrival latency threshold in seconds"
        )
    )

    parser.add_argument(
        "--update_type",
        type=str,
        default="fedopt",
        help=(
            "Aggregation type: fedopt, fedadam or fedavg"
        )
    )

    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.01,
        help=(
            "Server-side learning rate"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help=(
            "Output metrics directory"
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Initial seed
    # -------------------------------------------------------------

    set_seeds(
        data_seed
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(
        f"Execution started on target device: "
        f"{device}"
    )

    # =============================================================
    # LOAD CONFIGURATION
    # =============================================================

    with open(
        config_file,
        "r"
    ) as config_f:

        config = json.load(
            config_f
        )

    # -------------------------------------------------------------
    # Select participating clients
    # -------------------------------------------------------------

    devices_list = random.sample(
        config["devices_list"],
        network_size
    )

    # =============================================================
    # STEP 1:
    # LOAD RAW CLIENT DATA
    # =============================================================

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

        # ---------------------------------------------------------
        # Abnormal data
        # ---------------------------------------------------------

        if dev.get(
            "abnormal_data_path"
        ):

            abnormal_data_path = os.path.join(
                config["data_path"],
                dev["abnormal_data_path"]
            )

        else:

            abnormal_data_path = (
                normal_data_path
                .replace(
                    "normal",
                    "abnormal"
                )
            )

        try:

            abnormal_data = (
                load_data(
                    abnormal_data_path
                )
                .sample(
                    frac=1,
                    random_state=data_seed
                )
                .reset_index(
                    drop=True
                )
            )

        except Exception as exc:

            logging.warning(
                f"Abnormal data load failed for "
                f"{dev['name']}: {exc}"
            )

            abnormal_data = None

        # ---------------------------------------------------------
        # Split normal data
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Dedicated test_normal.csv
        # ---------------------------------------------------------

        test_normal_data_path = os.path.join(
            config["data_path"],
            dev["test_normal_data_path"]
        )

        test_normal_data = (
            load_data(
                test_normal_data_path
            )
            .sample(
                frac=1,
                random_state=data_seed
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Bootstrap subset
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Remaining local training data
        # ---------------------------------------------------------

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
                valid_normal_data,

            "test_normal_data":
                test_normal_data,

            "abnormal_data":
                abnormal_data,

            "sim_train_time":
                dev.get(
                    "simulated_training_time",
                    random.uniform(
                        1.0,
                        5.0
                    )
                ),

            "sim_comm_time":
                dev.get(
                    "simulated_comm_time",
                    random.uniform(
                        0.2,
                        1.5
                    )
                )
        })

    # =============================================================
    # STEP 2:
    # SERVER BOOTSTRAP DATASET
    # =============================================================

    bootstrap_server_dataframe = pd.concat(
        [
            client["bootstrap_data"]
            for client in raw_client_data
        ],
        ignore_index=True
    )

    logging.info(
        f"Initial server bootstrap dataset created "
        f"from {len(raw_client_data)} clients | "
        f"Samples: {len(bootstrap_server_dataframe)}"
    )

    # =============================================================
    # STEP 3:
    # COMMON PREPROCESSING
    # =============================================================

    data_processor = IoTDataProcessor(
        scaler="standard",
        use_log_transform=True,
        n_selected_features=target_num_features
    )

    processed_bootstrap_data, bootstrap_label = (
        data_processor.fit_transform(
            bootstrap_server_dataframe
        )
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

    selected_feature_indices = (
        data_processor.get_selected_features()
    )

    if selected_feature_indices is not None:

        logging.info(
            f"Selected feature indices: "
            f"{selected_feature_indices.tolist()}"
        )

    # =============================================================
    # STEP 4:
    # PROCESS EVERY CLIENT
    # =============================================================

    client_info = []

    for client in raw_client_data:

        processed_train_data, train_label = (
            data_processor.transform(
                client["train_data"],
                type="normal"
            )
        )

        processed_valid_data, valid_label = (
            data_processor.transform(
                client["valid_data"],
                type="normal"
            )
        )

        processed_test_normal, test_normal_label = (
            data_processor.transform(
                client["test_normal_data"],
                type="normal"
            )
        )

        if (
            client["abnormal_data"]
            is not None
        ):

            processed_test_abnormal, test_abnormal_label = (
                data_processor.transform(
                    client["abnormal_data"],
                    type="abnormal"
                )
            )

            test_data_combined = np.vstack(
                [
                    processed_test_normal,
                    processed_test_abnormal
                ]
            )

            test_label_combined = np.hstack(
                [
                    test_normal_label,
                    test_abnormal_label
                ]
            )

        else:

            test_data_combined = (
                processed_test_normal
            )

            test_label_combined = (
                test_normal_label
            )

        # ---------------------------------------------------------
        # Datasets
        # ---------------------------------------------------------

        train_dataset = IoTDataset(
            processed_train_data,
            train_label
        )

        valid_dataset = IoTDataset(
            processed_valid_data,
            valid_label
        )

        test_dataset = IoTDataset(
            test_data_combined,
            test_label_combined
        )

        # ---------------------------------------------------------
        # DataLoaders
        # ---------------------------------------------------------

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

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False
        )

        # ---------------------------------------------------------
        # Output directory
        # ---------------------------------------------------------

        client_dir = os.path.join(
            args.output_dir,
            client["device"]
        )

        os.makedirs(
            client_dir,
            exist_ok=True
        )

        client_info.append({

            "device":
                client["device"],

            "save_dir":
                client_dir,

            "train_loader":
                train_loader,

            "valid_loader":
                valid_loader,

            "test_loader":
                test_loader,

            "sim_train_time":
                client["sim_train_time"],

            "sim_comm_time":
                client["sim_comm_time"]
        })

    # =============================================================
    # STEP 5:
    # SERVER VALIDATION DATASET
    # =============================================================

    server_val_dataset = ConcatDataset(
        [
            client["valid_loader"].dataset
            for client in client_info
        ]
    )

    server_val_loader = DataLoader(
        dataset=server_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # =============================================================
    # STEP 6:
    # SERVER TEST DATASET
    # =============================================================

    server_test_dataset = ConcatDataset(
        [
            client["test_loader"].dataset
            for client in client_info
        ]
    )

    server_test_loader = DataLoader(
        dataset=server_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()

    all_experiment_results = {}

    # =============================================================
    # MODEL TYPES
    # =============================================================

    model_types = [
        "hybrid",
        "autoencoder"
    ]

    # =============================================================
    # EXPERIMENT LOOP
    # =============================================================

    for model_type in model_types:

        logging.info(
            "\n"
            "====================================================\n"
            f"STARTING EXPERIMENTS FOR MODEL TYPE: "
            f"{model_type.upper()}\n"
            "===================================================="
        )

        all_experiment_results[
            model_type
        ] = []

        # =========================================================
        # MULTIPLE RUNS
        # =========================================================

        for run in range(
            num_runs
        ):

            run_seed = (
                (run + 1)
                *
                10000
            )

            set_seeds(
                run_seed
            )

            logging.info(
                f"--- Starting Execution "
                f"Run {run + 1}/{num_runs} "
                f"(Seed: {run_seed}) ---"
            )

            # =====================================================
            # STEP 7:
            # INITIAL GLOBAL MODEL
            # =====================================================

            if model_type == "hybrid":

                global_model = (
                    Shrink_Autoencoder(
                        input_dim=actual_dim_features,
                        shrink_dim=shrink_dim,
                        threshold=threshold_val
                    )
                )

            else:

                global_model = (
                    Autoencoder(
                        input_dim=actual_dim_features,
                        hidden_neus=32,
                        latent_dim=shrink_dim,
                        output_dim=actual_dim_features,
                        use_sigmoid=False
                    )
                )

            global_model.to(
                device
            )

            # =====================================================
            # INITIAL SERVER TRAINING
            # =====================================================

            bootstrap_dataset = IoTDataset(
                processed_bootstrap_data,
                bootstrap_label
            )

            bootstrap_loader = DataLoader(
                dataset=bootstrap_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )

            logging.info(
                "[Initial Global Training] "
                f"Training global {model_type} "
                f"model using server bootstrap dataset..."
            )

            initial_trainer = ClientTrainer(
                model=global_model,
                client_id="SERVER_INIT",
                train_loader=bootstrap_loader,
                epoch=initial_epochs,
                lr_rate=lr_rate,
                update_type=args.update_type,
                device=str(device),
                save_dir=os.path.join(
                    args.output_dir,
                    "server_initial"
                ),
                kl_weight=vae_kl_weight
            )

            initial_trainer.run(
                bootstrap_loader,
                server_val_loader
            )

            global_model.load_state_dict(
                initial_trainer.get_parameters()
            )

            logging.info(
                "[Initial Global Training] "
                "Initial global model weights created successfully."
            )

            # =====================================================
            # FEDOPT AGGREGATOR
            # =====================================================

            global_aggregator = GlobalAggregator(
                model=global_model,
                update_type=args.update_type,
                server_lr=args.server_lr,
                max_server_update_norm=1.0
            )

            # =====================================================
            # SECURITY BUFFER
            # =====================================================

            sec_buffer_tracker = SecurityBuffer(
                global_model=global_model,
                window_size=5,
                latency_threshold=args.latency_threshold,
                alpha=0.2,
                beta=0.01
            )

            # =====================================================
            # PREVIOUS SECONDARY UPDATES
            # =====================================================

            carryover_updates = []

            # =====================================================
            # ROUND HISTORY
            # =====================================================

            run_round_history = []

            # =====================================================
            # CLIENT-WISE AUC HISTORY
            # =====================================================

            client_auc_history = []

            # =====================================================
            # TRAINING ROUNDS
            # =====================================================

            for round_idx in range(
                num_rounds
            ):

                round_start_time = time.time()

                round_number = (
                    round_idx + 1
                )

                logging.info(
                    f"[Run {run + 1} | Model {model_type}] "
                    f"--- Round {round_number}/{num_rounds} ---"
                )

                # -------------------------------------------------
                # Baseline global validation MSE
                # -------------------------------------------------

                global_mse = evaluate_global_mse(
                    global_aggregator.model,
                    server_val_loader,
                    device=device
                )

                logging.info(
                    f"[Round {round_number}] "
                    f"Pre-Aggregation Global Val MSE: "
                    f"{global_mse:.6f}"
                )

                incoming_updates = []

                # =================================================
                # CLIENT LOCAL TRAINING
                # =================================================

                for client in client_info:

                    c_start = time.time()

                    device_trainer = ClientTrainer(
                        model=global_aggregator.model,
                        client_id=client["device"],
                        save_dir=client["save_dir"],
                        epoch=epoch,
                        lr_rate=lr_rate,
                        update_type=args.update_type,
                        device=str(device),
                        kl_weight=vae_kl_weight
                    )

                    device_trainer.run(
                        client["train_loader"],
                        client["valid_loader"]
                    )

                    compute_time = (
                        time.time()
                        -
                        c_start
                    )

                    local_train_time = (
                        device_trainer.train_time
                    )

                    total_arrival_latency = (
                        compute_time
                        +
                        client["sim_comm_time"]
                    )

                    raw_weights = copy.deepcopy(
                        device_trainer.get_parameters()
                    )

                    incoming_updates.append({

                        "client_id":
                            client["device"],

                        "weights":
                            raw_weights,

                        "arrival_time":
                            total_arrival_latency,

                        "train_time":
                            local_train_time,

                        "dataset_size":
                            device_trainer.dataset_size,

                        "val_mse_list":
                            copy.deepcopy(
                                device_trainer.val_mse_list
                            ),

                        "val_loss":
                            device_trainer.val_loss,

                        "val_variance":
                            device_trainer.val_loss_variance,

                        "train_loss":
                            device_trainer.train_loss,

                        "reconstruction_loss":
                            device_trainer.reconstruction_loss,

                        "kl_loss":
                            device_trainer.kl_loss
                    })

                    logging.info(
                        f"[Client {client['device']}] "
                        f"Update prepared | "
                        f"Dataset: "
                        f"{device_trainer.dataset_size} | "
                        f"Train Time: "
                        f"{local_train_time:.2f}s | "
                        f"Arrival Latency: "
                        f"{total_arrival_latency:.2f}s | "
                        f"Val MSE: "
                        f"{device_trainer.val_loss:.6f}"
                    )

                # =================================================
                # SECURITY BUFFER
                # =================================================

                ready_updates = (
                    sec_buffer_tracker.collect_current_round_updates(
                        incoming_updates=incoming_updates,
                        global_model=global_aggregator.model,
                        val_loader=server_val_loader,
                        criterion=criterion,
                        global_mse=global_mse,
                        device=device
                    )
                )

                # -------------------------------------------------
                # Direct updates
                # -------------------------------------------------

                for update in ready_updates:

                    update["weight_factor"] = 1.0

                    update["aggregation_route"] = (
                        "DIRECT"
                    )

                # =================================================
                # SECONDARY / BUFFERED UPDATES
                # =================================================

                secondary_updates_for_next_round = []

                remaining_buffer = []

                for buffered_update in (
                    sec_buffer_tracker.buffer
                ):

                    client_id = buffered_update.get(
                        "client_id",
                        "unknown"
                    )

                    if buffered_update.get(
                        "validation_required",
                        False
                    ):

                        (
                            weight_factor,
                            server_val_mse,
                            validation_route
                        ) = validate_secondary_update(
                            update=buffered_update,
                            global_model=global_aggregator.model,
                            server_val_loader=server_val_loader,
                            current_global_mse=global_mse,
                            device=device
                        )

                        buffered_update[
                            "server_validation_mse"
                        ] = server_val_mse

                        buffered_update[
                            "weight_factor"
                        ] = weight_factor

                        buffered_update[
                            "aggregation_route"
                        ] = validation_route

                        if (
                            validation_route
                            ==
                            "SECONDARY_ACCEPTED"
                        ):

                            secondary_updates_for_next_round.append(
                                buffered_update
                            )

                            logging.info(
                                f"[Secondary Validation] "
                                f"Client {client_id} ACCEPTED | "
                                f"Server Val MSE: "
                                f"{server_val_mse:.6f} | "
                                f"Weight Factor: "
                                f"{weight_factor:.3f} | "
                                f"Scheduled for next round"
                            )

                        else:

                            secondary_updates_for_next_round.append(
                                buffered_update
                            )

                            logging.warning(
                                f"[Secondary Validation] "
                                f"Client {client_id} QUARANTINE | "
                                f"Server Val MSE: "
                                f"{server_val_mse:.6f} | "
                                f"Weight Factor: "
                                f"{weight_factor:.3f}"
                            )

                    else:

                        remaining_buffer.append(
                            buffered_update
                        )

                sec_buffer_tracker.buffer = (
                    remaining_buffer
                )

                # =================================================
                # CARRYOVER UPDATES
                # =================================================

                aggregation_updates = []

                if carryover_updates:

                    for update in carryover_updates:

                        aggregation_updates.append(
                            update
                        )

                aggregation_updates.extend(
                    ready_updates
                )

                carryover_updates = (
                    secondary_updates_for_next_round
                )

                # =================================================
                # FEDOPT AGGREGATION
                # =================================================

                if aggregation_updates:

                    global_aggregator.aggregate(
                        client_updates=aggregation_updates
                    )

                else:

                    logging.warning(
                        f"[Round {round_number}] "
                        "No updates available for aggregation."
                    )

                # =================================================
                # ROUND EVALUATION
                # =================================================

                round_duration = (
                    time.time()
                    -
                    round_start_time
                )

                # -------------------------------------------------
                # Post aggregation global validation MSE
                # -------------------------------------------------

                post_eval_mse = (
                    evaluate_global_mse(
                        global_aggregator.model,
                        server_val_loader,
                        device=device
                    )
                )

                # -------------------------------------------------
                # Global combined test evaluation
                # -------------------------------------------------

                global_metrics = (
                    evaluate_anomaly_detection(
                        global_aggregator.model,
                        server_test_loader,
                        device=device
                    )
                )

                # =================================================
                # NEW:
                # CLIENT-WISE GLOBAL MODEL EVALUATION
                # =================================================
                #
                # VERY IMPORTANT:
                #
                # This happens AFTER FedOpt aggregation.
                #
                # Therefore every client evaluates the SAME
                # CURRENT GLOBAL MODEL.
                #
                # Each client uses:
                #
                #     own test_normal.csv
                #              +
                #     own abnormal.csv
                #
                # Result:
                #
                #     Client-1 AUC
                #     Client-2 AUC
                #     ...
                #     Client-10 AUC
                #
                # =================================================

                client_wise_metrics = (
                    evaluate_clientwise_auc(
                        global_aggregator.model,
                        client_info,
                        device=device
                    )
                )

                # -------------------------------------------------
                # Store round client AUCs
                # -------------------------------------------------

                round_client_auc = {}

                for client_id, metrics in (
                    client_wise_metrics.items()
                ):

                    round_client_auc[
                        client_id
                    ] = metrics["auc"]

                    client_auc_history.append({

                        "run":
                            run + 1,

                        "model_type":
                            model_type,

                        "round":
                            round_number,

                        "client":
                            client_id,

                        "auc":
                            metrics["auc"],

                        "precision":
                            metrics["precision"],

                        "recall":
                            metrics["recall"],

                        "f1":
                            metrics["f1"]
                    })

                # -------------------------------------------------
                # Mean client AUC
                # -------------------------------------------------

                mean_client_auc = (
                    float(
                        np.mean(
                            list(
                                round_client_auc.values()
                            )
                        )
                    )
                    if round_client_auc
                    else 0.0
                )

                # =================================================
                # ROUND RECORD
                # =================================================

                round_record = {

                    "round":
                        round_number,

                    "global_val_mse":
                        post_eval_mse,

                    "test_precision":
                        global_metrics["precision"],

                    "test_recall":
                        global_metrics["recall"],

                    "test_f1":
                        global_metrics["f1_score"],

                    "test_auc":
                        global_metrics["auc_roc"],

                    # -------------------------------------------------
                    # NEW CLIENT-WISE AUC RESULTS
                    # -------------------------------------------------

                    "client_wise_auc":
                        round_client_auc,

                    "mean_client_auc":
                        mean_client_auc,

                    "round_duration":
                        round_duration,

                    "direct_updates":
                        len(ready_updates),

                    "aggregated_updates":
                        len(aggregation_updates),

                    "secondary_updates":
                        len(
                            secondary_updates_for_next_round
                        ),

                    "buffered_updates":
                        len(
                            sec_buffer_tracker.buffer
                        ),

                    "accepted_updates":
                        len(aggregation_updates)
                }

                run_round_history.append(
                    round_record
                )

                logging.info(
                    f"[Round {round_number} Finished] "
                    f"Val MSE: {post_eval_mse:.6f} | "
                    f"Global F1: {global_metrics['f1_score']:.4f} | "
                    f"Global AUC: {global_metrics['auc_roc']:.4f} | "
                    f"Mean Client AUC: {mean_client_auc:.4f} | "
                    f"Direct: {len(ready_updates)} | "
                    f"Aggregated: {len(aggregation_updates)} | "
                    f"Secondary: "
                    f"{len(secondary_updates_for_next_round)} | "
                    f"Buffered: "
                    f"{len(sec_buffer_tracker.buffer)} | "
                    f"Duration: {round_duration:.2f}s"
                )

            # =====================================================
            # SAVE RUN RESULTS
            # =====================================================

            all_experiment_results[
                model_type
            ].append(
                run_round_history
            )

            # =====================================================
            # SAVE CLIENT-WISE AUC CSV FOR THIS RUN
            # =====================================================

            client_auc_df = pd.DataFrame(
                client_auc_history
            )

            client_auc_csv_path = os.path.join(
                args.output_dir,
                f"{model_type}_run{run + 1}_client_auc_history.csv"
            )

            client_auc_df.to_csv(
                client_auc_csv_path,
                index=False
            )

            logging.info(
                f"Saved client-wise AUC history to: "
                f"{client_auc_csv_path}"
            )

            # =====================================================
            # FINAL ROUND CLIENT-WISE AUC
            # =====================================================

            if run_round_history:

                final_client_auc = (
                    run_round_history[-1]
                    .get(
                        "client_wise_auc",
                        {}
                    )
                )

                logging.info(
                    "===================================================="
                )

                logging.info(
                    f"[FINAL CLIENT-WISE AUC] "
                    f"Model: {model_type.upper()} | "
                    f"Run: {run + 1}"
                )

                for client_id, auc_value in (
                    final_client_auc.items()
                ):

                    logging.info(
                        f"{client_id}: "
                        f"AUC = {auc_value:.4f}"
                    )

                logging.info(
                    "===================================================="
                )

            # =====================================================
            # SAVE CHECKPOINT
            # =====================================================

            checkpoint_path = os.path.join(
                args.output_dir,
                f"{model_type}_run{run + 1}_checkpoint.pth"
            )

            torch.save(
                global_aggregator.model.state_dict(),
                checkpoint_path
            )

            logging.info(
                f"Saved run checkpoint to: "
                f"{checkpoint_path}"
            )

    # =============================================================
    # SUMMARY STATISTICS
    # =============================================================

    summary_report = {}

    for m_type in (
        all_experiment_results
    ):

        final_f1_scores = [
            run_data[-1][
                "test_f1"
            ]
            for run_data in (
                all_experiment_results[
                    m_type
                ]
            )
            if run_data
        ]

        final_auc_scores = [
            run_data[-1][
                "test_auc"
            ]
            for run_data in (
                all_experiment_results[
                    m_type
                ]
            )
            if run_data
        ]

        final_mse_scores = [
            run_data[-1][
                "global_val_mse"
            ]
            for run_data in (
                all_experiment_results[
                    m_type
                ]
            )
            if run_data
        ]

        # ---------------------------------------------------------
        # Final client-wise AUC from every run
        # ---------------------------------------------------------

        final_client_auc_runs = []

        for run_data in (
            all_experiment_results[
                m_type
            ]
        ):

            if run_data:

                final_client_auc_runs.append(
                    run_data[-1].get(
                        "client_wise_auc",
                        {}
                    )
                )

        summary_report[
            m_type
        ] = {

            "mean_f1":
                float(
                    np.mean(
                        final_f1_scores
                    )
                )
                if final_f1_scores
                else 0.0,

            "std_f1":
                float(
                    np.std(
                        final_f1_scores
                    )
                )
                if final_f1_scores
                else 0.0,

            "mean_auc":
                float(
                    np.mean(
                        final_auc_scores
                    )
                )
                if final_auc_scores
                else 0.0,

            "std_auc":
                float(
                    np.std(
                        final_auc_scores
                    )
                )
                if final_auc_scores
                else 0.0,

            "mean_mse":
                float(
                    np.mean(
                        final_mse_scores
                    )
                )
                if final_mse_scores
                else 0.0,

            "std_mse":
                float(
                    np.std(
                        final_mse_scores
                    )
                )
                if final_mse_scores
                else 0.0,

            # -----------------------------------------------------
            # FINAL CLIENT-WISE AUC
            # -----------------------------------------------------

            "final_client_wise_auc":
                final_client_auc_runs
        }

    # =============================================================
    # SAVE RESULTS
    # =============================================================

    summary_report_path = os.path.join(
        args.output_dir,
        "experiment_execution_results.json"
    )

    output_json_struct = {

        "summary":
            summary_report,

        "detailed_runs":
            all_experiment_results,

        "preprocessing": {

            "original_features":
                int(
                    bootstrap_server_dataframe.shape[1]
                ),

            "selected_features":
                int(
                    actual_dim_features
                ),

            "use_log_transform":
                True,

            "feature_selection":
                "Unsupervised variance-based selection",

            "scaler":
                "standard"
        },

        "vae": {

            "hidden_dimension":
                32,

            "latent_dimension":
                shrink_dim,

            "kl_weight":
                vae_kl_weight
        },

        "asynchronous_security": {

            "latency_threshold":
                args.latency_threshold,

            "bootstrap_fraction":
                bootstrap_fraction,

            "min_weight_factor":
                minimum_weight_factor,

            "quarantine_weight_factor":
                quarantine_weight_factor,

            "validation_rejection_ratio":
                validation_rejection_ratio
        },

        # ---------------------------------------------------------
        # NEW:
        # Client-wise evaluation description
        # ---------------------------------------------------------

        "client_wise_evaluation": {

            "evaluation_stage":
                "After FedOpt aggregation",

            "model_used":
                "Current global model",

            "test_data":
                "Each client's test_normal.csv + abnormal.csv",

            "metric":
                "ROC-AUC",

            "clients_per_round":
                network_size,

            "rounds":
                num_rounds
        }
    }

    with open(
        summary_report_path,
        "w"
    ) as f:

        json.dump(
            output_json_struct,
            f,
            indent=4
        )

    # =============================================================
    # FINAL CLIENT AUC CSV
    # =============================================================

    all_client_auc_df = []

    for m_type in all_experiment_results:

        for run_index, run_data in enumerate(
            all_experiment_results[
                m_type
            ],
            start=1
        ):

            for round_data in run_data:

                client_auc_data = (
                    round_data.get(
                        "client_wise_auc",
                        {}
                    )
                )

                for client_id, auc_value in (
                    client_auc_data.items()
                ):

                    all_client_auc_df.append({

                        "model_type":
                            m_type,

                        "run":
                            run_index,

                        "round":
                            round_data["round"],

                        "client":
                            client_id,

                        "auc":
                            auc_value
                    })

    if all_client_auc_df:

        all_client_auc_df = pd.DataFrame(
            all_client_auc_df
        )

        final_client_auc_csv = os.path.join(
            args.output_dir,
            "all_client_wise_auc.csv"
        )

        all_client_auc_df.to_csv(
            final_client_auc_csv,
            index=False
        )

        logging.info(
            f"All client-wise AUC results saved to: "
            f"{final_client_auc_csv}"
        )

    # =============================================================
    # COMPLETED
    # =============================================================

    logging.info(
        "\n"
        "====================================================\n"
        "PIPELINE EXECUTION COMPLETED\n"
        "===================================================="
    )

    logging.info(
        f"Results Summary saved successfully to: "
        f"{summary_report_path}"
    )
