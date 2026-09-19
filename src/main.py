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
    Initial server model training
        ↓
    Federated rounds
        ↓
    Local training
        ↓
    Timing attack manipulation
        ↓
    SecurityBuffer routing
        ↓
    Direct / Secondary / Quarantine
        ↓
    FedOpt aggregation
        ↓
    Global evaluation
        ↓
    Client-wise evaluation
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

# ============================================================
# COMET ML
# ============================================================
import comet_ml

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from torch.utils.data import DataLoader, ConcatDataset

from DataLoader.dataloader import (
    load_data,
    IoTDataset,
    IoTDataProcessor
)

from Trainer.client_trainer import ClientTrainer
from Trainer.global_aggregator import GlobalAggregator

from Trainer.malicious_update_experiment2 import manipulate_update

from Trainer.security_buffer import SecurityBuffer

from Models.shrink_autoencoder import Shrink_Autoencoder
from Models.autoencoder import Autoencoder


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# HYPERPARAMETERS
# ============================================================

num_participants = 1.0

epoch = 15
num_rounds = 15

lr_rate = 1e-5

shrink_dim = 16
threshold_val = 0.2

network_size = 10

data_seed = 1234

num_runs = 5

batch_size = 64

target_num_features = 64

bootstrap_fraction = 0.10

initial_epochs = 1

vae_kl_weight = 0.0001


# ============================================================
# TIMING ATTACK CONFIGURATION
# ============================================================

TIMING_ATTACK_CLIENT = "Client-3"

TIMING_ATTACK_START_ROUND = 3


# ============================================================
# SECURITY BUFFER WEIGHT FACTORS
# ============================================================

direct_weight_factor = 1.0

secondary_weight_factor = 0.7

quarantine_weight_factor = 0.3

minimum_weight_factor = 0.10


# ============================================================
# VALIDATION REJECTION
# ============================================================

validation_rejection_ratio = 4.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_seeds(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# RECONSTRUCTED OUTPUT
# ============================================================

def extract_reconstructed_output(model_output):

    if isinstance(model_output, tuple):

        return model_output[0]

    if isinstance(model_output, dict):

        if "reconstruction" in model_output:

            return model_output["reconstruction"]

        if "reconstructed" in model_output:

            return model_output["reconstructed"]

    return model_output


# ============================================================
# GLOBAL MSE EVALUATION
# ============================================================

def evaluate_global_mse(model, loader, device):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    criterion = nn.MSELoss(reduction="sum")

    with torch.no_grad():

        for batch in loader:

            if isinstance(batch, (list, tuple)):

                x = batch[0]

            else:

                x = batch

            x = x.to(device).float()

            output = model(x)

            reconstructed = extract_reconstructed_output(output)

            loss = criterion(reconstructed, x)

            total_loss += loss.item()

            total_samples += x.size(0)

    if total_samples == 0:

        return float("inf")

    return total_loss / total_samples


# ============================================================
# RECONSTRUCTION ERRORS
# ============================================================

def compute_reconstruction_errors(model, loader, device):

    model.eval()

    errors = []

    with torch.no_grad():

        for batch in loader:

            if isinstance(batch, (list, tuple)):

                x = batch[0]

            else:

                x = batch

            x = x.to(device).float()

            output = model(x)

            reconstructed = extract_reconstructed_output(output)

            sample_errors = torch.mean(
                (reconstructed - x) ** 2,
                dim=1
            )

            errors.extend(
                sample_errors.detach()
                .cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(errors)


# ============================================================
# ANOMALY DETECTION EVALUATION
# ============================================================

def evaluate_anomaly_detection(
    model,
    normal_loader,
    abnormal_loader,
    device
):

    normal_errors = compute_reconstruction_errors(
        model,
        normal_loader,
        device
    )

    abnormal_errors = compute_reconstruction_errors(
        model,
        abnormal_loader,
        device
    )

    all_errors = np.concatenate(
        [normal_errors, abnormal_errors]
    )

    labels = np.concatenate(
        [
            np.zeros(len(normal_errors)),
            np.ones(len(abnormal_errors))
        ]
    )

    # Threshold from normal reconstruction errors
    threshold = np.percentile(
        normal_errors,
        95
    )

    predictions = (
        all_errors >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            labels,
            all_errors
        )

    except Exception:

        auc = 0.5

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "auc_roc": auc,

        "threshold": threshold,

        "tp": int(tp),

        "fp": int(fp),

        "tn": int(tn),

        "fn": int(fn)
    }


# ============================================================
# CLIENT-WISE AUC
# ============================================================

def evaluate_clientwise_auc(
    model,
    client_data,
    device
):

    model.eval()

    results = {}

    for client in client_data:

        client_id = client["device"]

        normal_test_loader = client[
            "normal_test_loader"
        ]

        abnormal_test_loader = client[
            "abnormal_test_loader"
        ]

        try:

            metrics = evaluate_anomaly_detection(
                model,
                normal_test_loader,
                abnormal_test_loader,
                device
            )

            results[client_id] = {

                "auc": metrics["auc_roc"],

                "precision": metrics["precision"],

                "recall": metrics["recall"],

                "f1": metrics["f1_score"]
            }

        except Exception as e:

            logger.warning(
                f"Client-wise evaluation failed for "
                f"{client_id}: {e}"
            )

            results[client_id] = {

                "auc": 0.5,

                "precision": 0.0,

                "recall": 0.0,

                "f1": 0.0
            }

    return results


# ============================================================
# DELAYED UPDATE VALIDATION
# ============================================================

def validate_delayed_update(
    model,
    current_global_mse,
    delayed_update,
    server_validation_loader,
    device,
    rejection_ratio
):

    candidate_model = copy.deepcopy(model)

    try:

        candidate_model.load_state_dict(
            delayed_update["weights"]
        )

    except Exception as e:

        logger.error(
            f"Failed to load delayed update: {e}"
        )

        return False

    candidate_mse = evaluate_global_mse(
        candidate_model,
        server_validation_loader,
        device
    )

    rejection_limit = (
        current_global_mse *
        rejection_ratio
    )

    accepted = (
        candidate_mse <= rejection_limit
    )

    logger.info(
        f"Delayed update validation | "
        f"Candidate MSE={candidate_mse:.6f} | "
        f"Current Global MSE={current_global_mse:.6f} | "
        f"Limit={rejection_limit:.6f} | "
        f"Accepted={accepted}"
    )

    return accepted


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--latency_threshold",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--update_type",
        type=str,
        default="fedopt"
    )

    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results"
    )

    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Using device: {device}"
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    config_path = (
        "/content/fedmse/"
        "Configuration/"
        "scen2-nba-iot-10clients.json"
    )

    with open(
        config_path,
        "r"
    ) as f:

        config = json.load(f)


    # ========================================================
    # LOAD CLIENTS
    # ========================================================

    all_clients = load_data(
        config
    )


    # ========================================================
    # RANDOMLY SELECT CLIENTS
    # ========================================================

    random.seed(data_seed)

    selected_clients = random.sample(
        all_clients,
        network_size
    )


    logger.info(
        f"Selected {len(selected_clients)} clients"
    )


    # ========================================================
    # CLIENT DATA PREPARATION
    # ========================================================

    client_data = []

    bootstrap_frames = []


    for client in selected_clients:

        client_id = client["device"]

        logger.info(
            f"Preparing {client_id}"
        )


        normal_data = client[
            "normal"
        ]

        abnormal_data = client[
            "abnormal"
        ]


        # ----------------------------------------------------
        # SHUFFLE NORMAL DATA
        # ----------------------------------------------------

        normal_data = normal_data.sample(
            frac=1.0,
            random_state=data_seed
        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # SPLIT NORMAL DATA
        # ----------------------------------------------------

        n_total = len(normal_data)

        train_end = int(
            n_total * 0.40
        )

        val_end = int(
            n_total * 0.50
        )


        train_normal = normal_data[
            :train_end
        ].copy()

        validation_normal = normal_data[
            train_end:val_end
        ].copy()


        # ----------------------------------------------------
        # TEST NORMAL
        # ----------------------------------------------------

        test_normal = client[
            "test_normal"
        ]


        # ----------------------------------------------------
        # BOOTSTRAP
        # ----------------------------------------------------

        bootstrap_size = max(
            1,
            int(
                len(train_normal)
                * bootstrap_fraction
            )
        )


        bootstrap_sample = train_normal[
            :bootstrap_size
        ].copy()


        local_train = train_normal[
            bootstrap_size:
        ].copy()


        bootstrap_frames.append(
            bootstrap_sample
        )


        # ====================================================
        # DATASET OBJECTS
        # ====================================================

        train_dataset = IoTDataset(
            local_train
        )

        validation_dataset = IoTDataset(
            validation_normal
        )

        normal_test_dataset = IoTDataset(
            test_normal
        )

        abnormal_test_dataset = IoTDataset(
            abnormal_data
        )


        # ====================================================
        # DATA LOADERS
        # ====================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        normal_test_loader = DataLoader(
            normal_test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        abnormal_test_loader = DataLoader(
            abnormal_test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


        # ====================================================
        # SAVE CLIENT INFORMATION
        # ====================================================

        client_data.append({

            "device": client_id,

            "train_loader": train_loader,

            "validation_loader": validation_loader,

            "normal_test_loader": normal_test_loader,

            "abnormal_test_loader": abnormal_test_loader,

            "dataset_size": len(local_train)
        })


    # ========================================================
    # SERVER BOOTSTRAP DATA
    # ========================================================

    server_bootstrap_df = pd.concat(
        bootstrap_frames,
        ignore_index=True
    )


    logger.info(
        f"Server bootstrap samples: "
        f"{len(server_bootstrap_df)}"
    )


    # ========================================================
    # RAW PIPELINE
    # ========================================================

    logger.info(
        "Using RAW feature pipeline: "
        "No log transform, no feature selection, "
        "no scaling."
    )


    actual_dim_features = (
        server_bootstrap_df.shape[1]
    )


    logger.info(
        f"Actual number of input features: "
        f"{actual_dim_features}"
    )


    # ========================================================
    # SERVER VALIDATION DATASET
    # ========================================================

    validation_datasets = []

    test_datasets = []


    for client in client_data:

        validation_datasets.append(
            IoTDataset(
                client["validation_loader"].dataset
            )
        )

        test_datasets.append(
            IoTDataset(
                client["normal_test_loader"].dataset
            )
        )


    # ========================================================
    # SERVER VALIDATION LOADER
    # ========================================================

    server_validation_dataset = ConcatDataset(
        validation_datasets
    )

    server_validation_loader = DataLoader(
        server_validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # SERVER TEST DATASET
    # ========================================================

    server_test_normal_dataset = ConcatDataset(
        test_datasets
    )

    server_test_normal_loader = DataLoader(
        server_test_normal_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # MODEL TYPES
    # ========================================================

    model_types = [
        "hybrid",
        "autoencoder"
    ]


    # ========================================================
    # EXPERIMENT RESULTS
    # ========================================================

    experiment_results = {

        "config": {

            "num_participants": num_participants,

            "epoch": epoch,

            "num_rounds": num_rounds,

            "lr_rate": lr_rate,

            "shrink_dim": shrink_dim,

            "threshold_val": threshold_val,

            "network_size": network_size,

            "batch_size": batch_size,

            "bootstrap_fraction":
                bootstrap_fraction,

            "latency_threshold":
                args.latency_threshold,

            "server_lr":
                args.server_lr,

            "update_type":
                args.update_type
        },

        "models": {}
    }


    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_type in model_types:

        logger.info(
            "\n"
            + "=" * 80
        )

        logger.info(
            f"MODEL TYPE: {model_type}"
        )

        logger.info(
            "=" * 80
        )


        model_results = []


        # ====================================================
        # MULTIPLE RUNS
        # ====================================================

        for run in range(num_runs):

            run_seed = (
                (run + 1) * 10000
            )

            set_seeds(
                run_seed
            )


            # =================================================
            # COMET ML EXPERIMENT
            # =================================================

            experiment = comet_ml.Experiment(
                project_name="security-buffer"
            )

            experiment.set_name(
                f"{model_type}_run_{run + 1}"
            )

            experiment.add_tags([
                model_type,
                "security-buffer",
                "timing-attack",
                "raw-no-preprocessing",
                "fedopt"
            ])


            experiment.log_parameters({

                "model_type":
                    model_type,

                "run":
                    run + 1,

                "seed":
                    run_seed,

                "epochs":
                    epoch,

                "rounds":
                    num_rounds,

                "learning_rate":
                    lr_rate,

                "shrink_dim":
                    shrink_dim,

                "threshold_val":
                    threshold_val,

                "network_size":
                    network_size,

                "batch_size":
                    batch_size,

                "bootstrap_fraction":
                    bootstrap_fraction,

                "initial_epochs":
                    initial_epochs,

                "vae_kl_weight":
                    vae_kl_weight,

                "latency_threshold":
                    args.latency_threshold,

                "update_type":
                    args.update_type,

                "server_lr":
                    args.server_lr,

                "direct_weight_factor":
                    direct_weight_factor,

                "secondary_weight_factor":
                    secondary_weight_factor,

                "quarantine_weight_factor":
                    quarantine_weight_factor,

                "minimum_weight_factor":
                    minimum_weight_factor,

                "validation_rejection_ratio":
                    validation_rejection_ratio,

                "timing_attack_client":
                    TIMING_ATTACK_CLIENT,

                "timing_attack_start_round":
                    TIMING_ATTACK_START_ROUND,

                "feature_pipeline":
                    "RAW_NO_PREPROCESSING",

                "features_passed_to_model":
                    actual_dim_features
            })


            # =================================================
            # INITIAL MODEL
            # =================================================

            if model_type == "hybrid":

                global_model = Shrink_Autoencoder(
                    input_dim=actual_dim_features,
                    shrink_dim=shrink_dim
                )

            else:

                global_model = Autoencoder(
                    input_dim=actual_dim_features
                )


            global_model = global_model.to(
                device
            )


            # =================================================
            # INITIAL SERVER TRAINING
            # =================================================

            initial_trainer = ClientTrainer(
                model=copy.deepcopy(global_model),
                train_loader=server_validation_loader,
                val_loader=server_validation_loader,
                device=device,
                epochs=initial_epochs,
                lr=lr_rate,
                kl_weight=vae_kl_weight
            )


            initial_trainer.run()


            global_model.load_state_dict(
                initial_trainer.model.state_dict()
            )


            # =================================================
            # GLOBAL AGGREGATOR
            # =================================================

            aggregator = GlobalAggregator(
                global_model,
                update_type=args.update_type,
                server_lr=args.server_lr
            )


            # =================================================
            # SECURITY BUFFER
            # =================================================

            sec_buffer_tracker = SecurityBuffer(

                window_size=5,

                latency_threshold=
                    args.latency_threshold,

                alpha=0.2,

                beta=0.01
            )


            # =================================================
            # CARRYOVER UPDATES
            # =================================================

            carryover_updates = []


            # =================================================
            # ROUND HISTORY
            # =================================================

            run_round_history = []


            # =================================================
            # CLIENT AUC HISTORY
            # =================================================

            client_auc_history = []


            # =================================================
            # FEDERATED ROUNDS
            # =================================================

            for round_number in range(
                1,
                num_rounds + 1
            ):

                round_start_time = time.time()


                logger.info(
                    "\n"
                    + "-" * 80
                )

                logger.info(
                    f"{model_type} | "
                    f"Run {run + 1}/{num_runs} | "
                    f"Round {round_number}/{num_rounds}"
                )

                logger.info(
                    "-" * 80
                )


                # =================================================
                # PRE-AGGREGATION GLOBAL MSE
                # =================================================

                pre_global_mse = evaluate_global_mse(
                    global_model,
                    server_validation_loader,
                    device
                )


                logger.info(
                    f"Pre-aggregation Global MSE: "
                    f"{pre_global_mse:.6f}"
                )


                # =================================================
                # RELEASE PREVIOUS CARRYOVER UPDATES
                # =================================================

                previous_carryover_count = len(
                    carryover_updates
                )


                aggregation_updates = []


                for delayed_update in carryover_updates:

                    delayed_update_copy = copy.deepcopy(
                        delayed_update
                    )


                    route_type = delayed_update_copy.get(
                        "route_type",
                        "Secondary"
                    )


                    if route_type == "Secondary":

                        factor = (
                            secondary_weight_factor
                        )

                    else:

                        factor = (
                            quarantine_weight_factor
                        )


                    original_weight = (
                        delayed_update_copy.get(
                            "weight",
                            1.0
                        )
                    )


                    delayed_update_copy[
                        "weight"
                    ] = max(
                        minimum_weight_factor,
                        original_weight * factor
                    )


                    aggregation_updates.append(
                        delayed_update_copy
                    )


                carryover_updates = []


                # =================================================
                # CURRENT ROUND UPDATES
                # =================================================

                current_round_updates = []


                # =================================================
                # CLIENT LOCAL TRAINING
                # =================================================

                for client in client_data:

                    client_id = client[
                        "device"
                    ]


                    logger.info(
                        f"Training {client_id}"
                    )


                    # ---------------------------------------------
                    # LOCAL MODEL
                    # ---------------------------------------------

                    local_model = copy.deepcopy(
                        global_model
                    )


                    # ---------------------------------------------
                    # CLIENT TRAINER
                    # ---------------------------------------------

                    device_trainer = ClientTrainer(

                        model=local_model,

                        train_loader=
                            client["train_loader"],

                        val_loader=
                            client["validation_loader"],

                        device=device,

                        epochs=epoch,

                        lr=lr_rate,

                        kl_weight=vae_kl_weight
                    )


                    # ---------------------------------------------
                    # TRAINING TIME
                    # ---------------------------------------------

                    compute_start = time.time()


                    device_trainer.run()


                    compute_time = (
                        time.time()
                        - compute_start
                    )


                    # ---------------------------------------------
                    # TRAINING TIME FROM TRAINER
                    # ---------------------------------------------

                    train_time = (
                        device_trainer.train_time
                    )


                    # ---------------------------------------------
                    # COMMUNICATION TIME
                    # ---------------------------------------------

                    communication_time = (
                        max(
                            0.0,
                            compute_time - train_time
                        )
                    )


                    # ---------------------------------------------
                    # TOTAL ARRIVAL LATENCY
                    # ---------------------------------------------

                    arrival_latency = (
                        compute_time
                        + communication_time
                    )


                    # ---------------------------------------------
                    # UPDATE
                    # ---------------------------------------------

                    update = {

                        "device":
                            client_id,

                        "weights":
                            copy.deepcopy(
                                device_trainer.model.state_dict()
                            ),

                        "weight":
                            float(
                                client["dataset_size"]
                            ),

                        "arrival_time":
                            arrival_latency,

                        "train_time":
                            train_time,

                        "dataset_size":
                            client["dataset_size"],

                        "val_mse_list":
                            getattr(
                                device_trainer,
                                "val_mse_list",
                                []
                            ),

                        "val_loss":
                            device_trainer.val_loss,

                        "val_variance":
                            getattr(
                                device_trainer,
                                "val_variance",
                                0.0
                            ),

                        "train_loss":
                            device_trainer.train_loss,

                        "reconstruction_loss":
                            getattr(
                                device_trainer,
                                "reconstruction_loss",
                                0.0
                            ),

                        "kl_loss":
                            getattr(
                                device_trainer,
                                "kl_loss",
                                0.0
                            )
                    }


                    # =================================================
                    # TIMING ATTACK
                    # =================================================

                    if (

                        client_id
                        ==
                        TIMING_ATTACK_CLIENT

                        and

                        round_number
                        >=
                        TIMING_ATTACK_START_ROUND

                    ):

                        update = manipulate_update(
                            update,
                            attack_type="timing"
                        )

                        logger.info(
                            f"{client_id}: "
                            f"TIMING ATTACK ACTIVE"
                        )

                    else:

                        update = manipulate_update(
                            update,
                            attack_type="none"
                        )


                    # =================================================
                    # COMET CLIENT METRICS
                    # =================================================

                    client_comet_metrics = {

                        f"{client_id}_train_loss":
                            float(
                                device_trainer.train_loss
                            ),

                        f"{client_id}_val_loss":
                            float(
                                device_trainer.val_loss
                            ),

                        f"{client_id}_reconstruction_loss":
                            float(
                                getattr(
                                    device_trainer,
                                    "reconstruction_loss",
                                    0.0
                                )
                            ),

                        f"{client_id}_kl_loss":
                            float(
                                getattr(
                                    device_trainer,
                                    "kl_loss",
                                    0.0
                                )
                            ),

                        f"{client_id}_train_time":
                            float(train_time),

                        f"{client_id}_compute_time":
                            float(compute_time),

                        f"{client_id}_arrival_latency":
                            float(arrival_latency)
                    }


                    experiment.log_metrics(
                        client_comet_metrics,
                        step=round_number
                    )


                    # =================================================
                    # SECURITY BUFFER COLLECTION
                    # =================================================

                    current_round_updates.append(
                        update
                    )


                # =================================================
                # SECURITY BUFFER ROUTING
                # =================================================

                routed_updates = (
                    sec_buffer_tracker
                    .collect_current_round_updates(
                        current_round_updates
                    )
                )


                # =================================================
                # DIRECT UPDATES
                # =================================================

                direct_current_updates = []


                for update in routed_updates:

                    route = update.get(
                        "route",
                        update.get(
                            "route_type",
                            "Direct"
                        )
                    )


                    if route == "Direct":

                        update_copy = copy.deepcopy(
                            update
                        )

                        update_copy[
                            "weight"
                        ] = max(
                            minimum_weight_factor,
                            update_copy.get(
                                "weight",
                                1.0
                            )
                            * direct_weight_factor
                        )

                        direct_current_updates.append(
                            update_copy
                        )


                # =================================================
                # ADD DIRECT UPDATES TO AGGREGATION
                # =================================================

                aggregation_updates.extend(
                    direct_current_updates
                )


                # =================================================
                # DELAYED UPDATE VALIDATION
                # =================================================

                secondary_accepted = 0

                quarantine_accepted = 0

                dropped_updates = 0

                secondary_updates_for_next_round = []

                quarantine_updates_for_next_round = []


                # =================================================
                # VALIDATE BUFFERED UPDATES
                # =================================================

                buffered_updates = (
                    sec_buffer_tracker.buffer
                )


                for delayed_update in buffered_updates:

                    route_type = delayed_update.get(
                        "route_type",
                        delayed_update.get(
                            "route",
                            "Secondary"
                        )
                    )


                    accepted = validate_delayed_update(

                        global_model,

                        pre_global_mse,

                        delayed_update,

                        server_validation_loader,

                        device,

                        validation_rejection_ratio
                    )


                    if accepted:

                        delayed_copy = copy.deepcopy(
                            delayed_update
                        )


                        if route_type == "Secondary":

                            delayed_copy[
                                "route_type"
                            ] = "Secondary"

                            secondary_updates_for_next_round.append(
                                delayed_copy
                            )

                            secondary_accepted += 1


                        else:

                            delayed_copy[
                                "route_type"
                            ] = "Quarantine"

                            quarantine_updates_for_next_round.append(
                                delayed_copy
                            )

                            quarantine_accepted += 1


                    else:

                        dropped_updates += 1


                # =================================================
                # CARRY OVER TO NEXT ROUND
                # =================================================

                carryover_updates.extend(
                    secondary_updates_for_next_round
                )

                carryover_updates.extend(
                    quarantine_updates_for_next_round
                )


                # =================================================
                # CLEAR CURRENT SECURITY BUFFER
                # =================================================

                sec_buffer_tracker.buffer.clear()


                # =================================================
                # FEDOPT AGGREGATION
                # =================================================

                if len(aggregation_updates) > 0:

                    global_model = aggregator.aggregate(
                        aggregation_updates
                    )

                    global_model = global_model.to(
                        device
                    )

                else:

                    logger.warning(
                        "No updates available "
                        "for aggregation."
                    )


                # =================================================
                # POST-AGGREGATION MSE
                # =================================================

                post_global_mse = evaluate_global_mse(
                    global_model,
                    server_validation_loader,
                    device
                )


                # =================================================
                # GLOBAL TEST METRICS
                # =================================================

                abnormal_test_loaders = [

                    client["abnormal_test_loader"]

                    for client in client_data
                ]


                normal_test_loaders = [

                    client["normal_test_loader"]

                    for client in client_data
                ]


                normal_test_dataset = ConcatDataset(
                    [
                        loader.dataset
                        for loader in normal_test_loaders
                    ]
                )


                abnormal_test_dataset = ConcatDataset(
                    [
                        loader.dataset
                        for loader in abnormal_test_loaders
                    ]
                )


                global_normal_test_loader = DataLoader(
                    normal_test_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )


                global_abnormal_test_loader = DataLoader(
                    abnormal_test_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )


                global_metrics = evaluate_anomaly_detection(

                    global_model,

                    global_normal_test_loader,

                    global_abnormal_test_loader,

                    device
                )


                # =================================================
                # CLIENT-WISE EVALUATION
                # =================================================

                client_wise_metrics = evaluate_clientwise_auc(

                    global_model,

                    client_data,

                    device
                )


                client_auc_history.append(
                    {
                        "round":
                            round_number,

                        "metrics":
                            client_wise_metrics
                    }
                )


                # =================================================
                # MEAN CLIENT AUC
                # =================================================

                if len(client_wise_metrics) > 0:

                    mean_client_auc = np.mean(
                        [
                            metrics["auc"]
                            for metrics
                            in client_wise_metrics.values()
                        ]
                    )

                else:

                    mean_client_auc = 0.5


                # =================================================
                # ROUND DURATION
                # =================================================

                round_duration = (
                    time.time()
                    - round_start_time
                )


                # =================================================
                # ROUND HISTORY
                # =================================================

                round_record = {

                    "round":
                        round_number,

                    "pre_global_mse":
                        pre_global_mse,

                    "post_global_mse":
                        post_global_mse,

                    "precision":
                        global_metrics["precision"],

                    "recall":
                        global_metrics["recall"],

                    "f1":
                        global_metrics["f1_score"],

                    "auc":
                        global_metrics["auc_roc"],

                    "threshold":
                        global_metrics["threshold"],

                    "tp":
                        global_metrics["tp"],

                    "fp":
                        global_metrics["fp"],

                    "tn":
                        global_metrics["tn"],

                    "fn":
                        global_metrics["fn"],

                    "mean_client_auc":
                        mean_client_auc,

                    "direct_updates":
                        len(
                            direct_current_updates
                        ),

                    "carryover_updates":
                        previous_carryover_count,

                    "aggregated_updates":
                        len(
                            aggregation_updates
                        ),

                    "secondary_accepted":
                        secondary_accepted,

                    "quarantine_accepted":
                        quarantine_accepted,

                    "dropped_updates":
                        dropped_updates,

                    "buffered_updates":
                        len(
                            buffered_updates
                        ),

                    "round_duration":
                        round_duration,

                    "client_metrics":
                        client_wise_metrics
                }


                run_round_history.append(
                    round_record
                )


                # =================================================
                # COMET ROUND-LEVEL METRICS
                # =================================================

                experiment.log_metrics({

                    "pre_global_mse":
                        float(pre_global_mse),

                    "global_val_mse":
                        float(post_global_mse),

                    "test_precision":
                        float(
                            global_metrics["precision"]
                        ),

                    "test_recall":
                        float(
                            global_metrics["recall"]
                        ),

                    "test_f1":
                        float(
                            global_metrics["f1_score"]
                        ),

                    "test_auc":
                        float(
                            global_metrics["auc_roc"]
                        ),

                    "anomaly_threshold":
                        float(
                            global_metrics["threshold"]
                        ),

                    "true_positive":
                        int(
                            global_metrics["tp"]
                        ),

                    "false_positive":
                        int(
                            global_metrics["fp"]
                        ),

                    "true_negative":
                        int(
                            global_metrics["tn"]
                        ),

                    "false_negative":
                        int(
                            global_metrics["fn"]
                        ),

                    "mean_client_auc":
                        float(mean_client_auc),

                    "direct_updates":
                        int(
                            len(
                                direct_current_updates
                            )
                        ),

                    "carryover_updates":
                        int(
                            previous_carryover_count
                        ),

                    "aggregated_updates":
                        int(
                            len(
                                aggregation_updates
                            )
                        ),

                    "secondary_accepted":
                        int(
                            secondary_accepted
                        ),

                    "quarantine_accepted":
                        int(
                            quarantine_accepted
                        ),

                    "dropped_updates":
                        int(
                            dropped_updates
                        ),

                    "buffered_updates":
                        int(
                            len(buffered_updates)
                        ),

                    "round_duration":
                        float(round_duration)
                },
                step=round_number
                )


                # =================================================
                # COMET CLIENT-WISE METRICS
                # =================================================

                client_auc_comet_metrics = {}


                for client_id, metrics in (
                    client_wise_metrics.items()
                ):

                    client_auc_comet_metrics.update({

                        f"{client_id}_auc":
                            float(
                                metrics["auc"]
                            ),

                        f"{client_id}_precision":
                            float(
                                metrics["precision"]
                            ),

                        f"{client_id}_recall":
                            float(
                                metrics["recall"]
                            ),

                        f"{client_id}_f1":
                            float(
                                metrics["f1"]
                            )
                    })


                experiment.log_metrics(
                    client_auc_comet_metrics,
                    step=round_number
                )


                # =================================================
                # LOGGING
                # =================================================

                logger.info(
                    f"Round {round_number} Results | "
                    f"Precision={global_metrics['precision']:.4f} | "
                    f"Recall={global_metrics['recall']:.4f} | "
                    f"F1={global_metrics['f1_score']:.4f} | "
                    f"AUC={global_metrics['auc_roc']:.4f}"
                )


                logger.info(
                    f"Global MSE | "
                    f"Pre={pre_global_mse:.6f} | "
                    f"Post={post_global_mse:.6f}"
                )


                logger.info(
                    f"Routing | "
                    f"Direct={len(direct_current_updates)} | "
                    f"Aggregated={len(aggregation_updates)} | "
                    f"Secondary={secondary_accepted} | "
                    f"Quarantine={quarantine_accepted} | "
                    f"Dropped={dropped_updates}"
                )


                logger.info(
                    f"Mean Client AUC="
                    f"{mean_client_auc:.4f}"
                )


            # ====================================================
            # FINAL RUN METRICS
            # ====================================================

            if len(run_round_history) > 0:

                final_record = (
                    run_round_history[-1]
                )


                final_f1 = (
                    final_record["f1"]
                )

                final_auc = (
                    final_record["auc"]
                )

                final_mse = (
                    final_record["post_global_mse"]
                )


            else:

                final_f1 = 0.0

                final_auc = 0.5

                final_mse = float("inf")


            # ====================================================
            # COMET FINAL RUN METRICS
            # ====================================================

            experiment.log_metrics({

                "final_f1":
                    float(final_f1),

                "final_auc":
                    float(final_auc),

                "final_global_mse":
                    float(final_mse),

                "final_mean_client_auc":
                    float(
                        run_round_history[-1][
                            "mean_client_auc"
                        ]
                    )
                    if run_round_history
                    else 0.5
            })


            # ====================================================
            # SAVE CLIENT-WISE AUC
            # ====================================================

            client_auc_rows = []


            for round_data in client_auc_history:

                round_number_value = (
                    round_data["round"]
                )


                for client_id, metrics in (
                    round_data["metrics"].items()
                ):

                    client_auc_rows.append({

                        "round":
                            round_number_value,

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


            client_auc_df = pd.DataFrame(
                client_auc_rows
            )


            client_auc_path = os.path.join(

                args.output_dir,

                f"{model_type}_run_{run + 1}"
                "_clientwise_auc.csv"
            )


            client_auc_df.to_csv(
                client_auc_path,
                index=False
            )


            # ====================================================
            # SAVE CHECKPOINT
            # ====================================================

            checkpoint_path = os.path.join(

                args.output_dir,

                f"{model_type}_run_{run + 1}"
                "_global_model.pt"
            )


            torch.save(

                global_model.state_dict(),

                checkpoint_path
            )


            # ====================================================
            # SAVE RUN RESULTS
            # ====================================================

            model_results.append({

                "run":
                    run + 1,

                "seed":
                    run_seed,

                "final_f1":
                    final_f1,

                "final_auc":
                    final_auc,

                "final_mse":
                    final_mse,

                "mean_client_auc":
                    (
                        run_round_history[-1][
                            "mean_client_auc"
                        ]
                        if run_round_history
                        else 0.5
                    ),

                "round_history":
                    run_round_history,

                "client_auc_history":
                    client_auc_history
            })


            # ====================================================
            # COMET END
            # ====================================================

            experiment.end()


            logger.info(
                f"Completed {model_type} "
                f"Run {run + 1}/{num_runs}"
            )


        # ========================================================
        # MODEL SUMMARY
        # ========================================================

        final_f1_values = [

            result["final_f1"]

            for result in model_results
        ]


        final_auc_values = [

            result["final_auc"]

            for result in model_results
        ]


        final_mse_values = [

            result["final_mse"]

            for result in model_results
        ]


        mean_client_auc_values = [

            result["mean_client_auc"]

            for result in model_results
        ]


        model_summary = {

            "final_f1_mean":
                float(
                    np.mean(final_f1_values)
                ),

            "final_f1_std":
                float(
                    np.std(final_f1_values)
                ),

            "final_auc_mean":
                float(
                    np.mean(final_auc_values)
                ),

            "final_auc_std":
                float(
                    np.std(final_auc_values)
                ),

            "final_mse_mean":
                float(
                    np.mean(final_mse_values)
                ),

            "final_mse_std":
                float(
                    np.std(final_mse_values)
                ),

            "mean_client_auc":
                float(
                    np.mean(
                        mean_client_auc_values
                    )
                ),

            "mean_client_auc_std":
                float(
                    np.std(
                        mean_client_auc_values
                    )
                ),

            "runs":
                model_results
        }


        experiment_results[
            "models"
        ][model_type] = model_summary


        # ========================================================
        # MODEL SUMMARY LOG
        # ========================================================

        logger.info(
            "\n"
            + "=" * 80
        )

        logger.info(
            f"SUMMARY - {model_type}"
        )

        logger.info(
            "=" * 80
        )

        logger.info(
            f"Final F1: "
            f"{model_summary['final_f1_mean']:.4f} "
            f"+/- "
            f"{model_summary['final_f1_std']:.4f}"
        )

        logger.info(
            f"Final AUC: "
            f"{model_summary['final_auc_mean']:.4f} "
            f"+/- "
            f"{model_summary['final_auc_std']:.4f}"
        )

        logger.info(
            f"Final MSE: "
            f"{model_summary['final_mse_mean']:.6f} "
            f"+/- "
            f"{model_summary['final_mse_std']:.6f}"
        )

        logger.info(
            f"Mean Client AUC: "
            f"{model_summary['mean_client_auc']:.4f} "
            f"+/- "
            f"{model_summary['mean_client_auc_std']:.4f}"
        )


    # ========================================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================================

    results_path = os.path.join(

        args.output_dir,

        "experiment_execution_results.json"
    )


    with open(
        results_path,
        "w"
    ) as f:

        json.dump(
            experiment_results,
            f,
            indent=4
        )


    # ========================================================
    # SAVE ALL CLIENT-WISE AUC
    # ========================================================

    all_client_auc_rows = []


    for model_type, model_summary in (
        experiment_results["models"].items()
    ):

        for run_result in model_summary["runs"]:

            run_number = (
                run_result["run"]
            )


            for round_data in (
                run_result[
                    "client_auc_history"
                ]
            ):

                round_number_value = (
                    round_data["round"]
                )


                for client_id, metrics in (
                    round_data[
                        "metrics"
                    ].items()
                ):

                    all_client_auc_rows.append({

                        "model":
                            model_type,

                        "run":
                            run_number,

                        "round":
                            round_number_value,

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


    all_client_auc_df = pd.DataFrame(
        all_client_auc_rows
    )


    all_client_auc_path = os.path.join(

        args.output_dir,

        "all_clientwise_auc.csv"
    )


    all_client_auc_df.to_csv(
        all_client_auc_path,
        index=False
    )


    # ========================================================
    # COMPLETION
    # ========================================================

    logger.info(
        "\n"
        + "=" * 80
    )

    logger.info(
        "EXPERIMENT COMPLETED"
    )

    logger.info(
        "=" * 80
    )

    logger.info(
        f"Results saved to: "
        f"{args.output_dir}"
    )

    logger.info(
        f"JSON: "
        f"{results_path}"
    )

    logger.info(
        f"Client-wise AUC: "
        f"{all_client_auc_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
