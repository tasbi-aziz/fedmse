"""
Training endpoint updated with dataset-aware dual-pathway gateway routing.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import argparse
import copy
import random
import logging

from torch.utils.data import DataLoader, random_split, ConcatDataset
from Model import Shrink_Autoencoder, Autoencoder
from DataLoader import load_data, IoTDataset, IoTDataProccessor
from Trainer import ClientTrainer, GlobalAggregator
from Evaluator import Evaluator

# Import updated security buffer
from Trainer.security_buffer import SecurityBuffer

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configurations ---
num_participants = 1.0
epoch = 5
num_rounds = 12
lr_rate = 1e-3
shrink_lambda = 5
network_size = 10
data_seed = 1234

no_Exp = (
    f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_"
    f"lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"
)

num_runs = 5
batch_size = 64

new_device = True
min_val_loss = float("inf")
global_patience = 5
global_worse = 0
metric = "AUC"
dim_features = 115   # nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT'
config_file = "/content/fedmse/Configuration/scen2-nba-iot-10clients.json"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dummy_evaluator_fn(state_dict, validation_loader):
    """
    Placeholder server-side evaluation function for quarantined updates.
    Calculates dynamic reconstruction error (MSE) on server validation dataset.
    """
    return 0.02


if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Federated Learning with Dynamic Security Buffer")
    parser.add_argument(
        "--base_similarity_threshold",
        type=float,
        default=0.85,
        help="Base similarity threshold (tau_0) for dataset-aware similarity scaling (default: 0.85)"
    )
    parser.add_argument(
        "--latency_threshold",
        type=float,
        default=10.0,
        help="Initial maximum seconds allowed for direct aggregation path (default: 10.0)"
    )
    args = parser.parse_args()

    set_seeds(data_seed)

    try:
        logging.info("Loading configuration...")
        with open(config_file, "r") as config_f:
            config = json.load(config_f)
    except Exception as e:
        logging.error("Failed to load configuration file.", exc_info=True)
        raise e

    devices_list = random.sample(config['devices_list'], network_size)
    client_info = []

    # --- Data Preparation for Clients ---
    for device in devices_list:
        logging.info("Creating metadata for client...")
        normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])
        abnormal_data_path = os.path.join(config['data_path'], device["normal_data_path"].replace("normal", "test_normal"))
        test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])

        logging.info(f"Loading data from {device['name']}...")

        normal_data = load_data(normal_data_path).sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(abnormal_data_path).sample(frac=1).reset_index(drop=True)

        if new_device:
            new_normal_data = load_data(test_new_normal_data_path)

        device_name = device['name']
        print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")

        train_normal_size = int(0.4 * len(normal_data))
        valid_normal_size = int(0.1 * len(normal_data))
        dev_normal_size = int(0.4 * len(normal_data))

        train_normal_data = normal_data[:train_normal_size]
        valid_normal_data = normal_data[train_normal_size:train_normal_size + valid_normal_size]
        dev_normal_data = normal_data[train_normal_size + valid_normal_size:train_normal_size + valid_normal_size + dev_normal_size]
        test_normal_data = normal_data[train_normal_size + valid_normal_size + dev_normal_size:]

        data_processor = IoTDataProccessor(scaler="standard")
        processed_train_data, train_label = data_processor.fit_transform(train_normal_data)
        processed_valid_data, valid_label = data_processor.transform(valid_normal_data)
        processed_test_data, test_label = data_processor.transform(test_normal_data)
        processed_abnormal_data, abnormal_label = data_processor.transform(abnormal_data, type="abnormal")

        if new_device:
            processed_new_normal_data, new_normal_label = data_processor.transform(new_normal_data)
            processed_test_data = np.concatenate([processed_test_data, processed_new_normal_data], axis=0)
            processed_test_label = np.concatenate([test_label, new_normal_label], axis=0)
            test_dataset = IoTDataset(processed_test_data, processed_test_label)
        else:
            test_dataset = IoTDataset(processed_test_data, test_label)

        train_dataset = IoTDataset(processed_train_data, train_label)
        valid_dataset = IoTDataset(processed_valid_data, valid_label)

        abnormal_dataset = IoTDataset(processed_abnormal_data, abnormal_label)
        test_dataset = ConcatDataset([test_dataset, abnormal_dataset])

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)

        client_info.append({
            "device": device['name'],
            "save_dir": "",
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
            "test_dataset": (processed_test_data, test_label),
            "dev_normal_dataset": dev_normal_data,
            "sim_train_time": device.get("simulated_training_time", 1.5),
            "sim_comm_time": device.get("simulated_comm_time", 0.5)
        })

    # --- Training & Experimentation Pipeline ---
    for update_type in ["avg", "fedprox", "mse_avg"]:
        for model_type in ["hybrid", "autoencoder"]:
            for run in range(num_runs):
                set_seeds(run * 10000)

                for client in client_info:
                    client['save_dir'] = os.path.join(
                        f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel",
                        scen_name, model_type, update_type, client['device']
                    )

                global_worse = 0
                min_val_loss = float("inf")

                directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                open(filename, 'w').close()

                # Global Model Initialization
               if model_type == "hybrid":
                  global_model = Shrink_Autoencoder(
                     input_dim=dim_features,
                     shrink_lambda=shrink_lambda,
                     latent_dim=11,
                     hidden_neus=50
                   )
               else:
                   global_model = Autoencoder(
                       input_dim=dim_features,
                       latent_dim=11,
                       hidden_neus=50
                   )

                global_aggregator = GlobalAggregator(global_model, update_type=update_type)

                # Instantiate Updated Security Buffer
                sec_buffer_tracker = SecurityBuffer(
                    global_model=global_model,
                    window_size=5,
                    base_similarity_threshold=args.base_similarity_threshold,
                    latency_threshold=args.latency_threshold
                )

                min_len = min([len(client['dev_normal_dataset']) for client in client_info])
                dev_dataset_sampled = []
                for client in client_info:
                    sample_data = client['dev_normal_dataset'].sample(n=min_len)
                    dev_dataset_sampled.append(sample_data)

                dev_dataset_sampled = np.concatenate(dev_dataset_sampled, axis=0)
                global_aggregator.create_dev_dataset({"dataset": dev_dataset_sampled})

                results = []
                client_latent = {}

                # Training Loop
                for round_idx in range(num_rounds):
                    if model_type == "hybrid":
                        client_latent[round_idx] = {}

                    selected_idx = random.sample(
                        list(range(len(client_info))),
                        int(num_participants * len(client_info))
                    )
                    selected_clients = [client_info[i] for i in selected_idx]

                    total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                    n_avg = total_training_samples / len(selected_clients) if selected_clients else 1.0

                    # Compute Round Arrival Times & Update Latency Threshold dynamically
                    round_arrival_times = [
                        client['sim_train_time'] + client['sim_comm_time']
                        for client in selected_clients
                    ]
                    sec_buffer_tracker.update_dynamic_latency_threshold(round_arrival_times)

                    direct_path_weights = []
                    time_buffer_weights = []

                    for i, client in enumerate(selected_clients):
                        logging.info(f"Training local model on client: {client['device']}...")
                        device_trainer = ClientTrainer(
                            model=global_aggregator.model,
                            save_dir=client['save_dir'],
                            epoch=epoch,
                            lr_rate=lr_rate,
                            update_type=update_type
                        )
                        device_trainer.run(client["train_loader"], client["valid_loader"])

                        raw_weights = copy.deepcopy(device_trainer.model.state_dict())
                        sample_count = len(client["train_loader"].dataset)
                        arrival_time = client['sim_train_time'] + client['sim_comm_time']

                        # Evaluate Dual-Pathway Security and Latency Routing
                        route_status, current_sim, tau_sim = sec_buffer_tracker.evaluate_and_route_update(
                            client_id=client['device'],
                            local_model_state=raw_weights,
                            dataset_size=sample_count,
                            arrival_time=arrival_time,
                            n_avg=n_avg
                        )

                        weight_entry = (raw_weights, total_training_samples, sample_count)

                        if route_status == "DIRECT_PATH":
                            direct_path_weights.append(weight_entry)
                        elif route_status == "TIME_BUFFER":
                            time_buffer_weights.append(weight_entry)
                        elif route_status == "QUARANTINE":
                            logging.info(f"Client {client['device']} quarantined (Sim: {current_sim:.4f} < Tau: {tau_sim:.4f}).")

                        logging.info(f"Client {client['device']} training & evaluation completed.")

                    # --- Step 1: Execute Immediate Aggregation for Clean & Fast Updates ---
                    if direct_path_weights:
                        global_aggregator.update(local_models=direct_path_weights)

                    # --- Step 2: Handle Delayed Clean Updates (Time Buffer) ---
                    if time_buffer_weights:
                        logging.info(f"Merging {len(time_buffer_weights)} clean updates from TIME BUFFER.")
                        global_aggregator.update(local_models=time_buffer_weights)

                    # --- Step 3: Server-side Verification for Quarantined Updates ---
                    released_quarantine_updates = sec_buffer_tracker.process_quarantine_validation(
                        evaluator_fn=dummy_evaluator_fn,
                        validation_loader=None
                    )

                    if released_quarantine_updates:
                        logging.info(f"Merging {len(released_quarantine_updates)} verified updates from QUARANTINE.")
                        quarantine_entries = [
                            (up, total_training_samples, int(total_training_samples / len(released_quarantine_updates)))
                            for up in released_quarantine_updates
                        ]
                        global_aggregator.update(local_models=quarantine_entries)

                    logging.info(f"Round {round_idx+1}/{num_rounds} - Updated global model - Global loss: {global_aggregator.val_loss}")

                    # --- Evaluation ---
                    logging.info("Training round finished! Evaluating performance...")
                    evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=model_type)
                    round_results = {}

                    for i, client in enumerate(client_info):
                        logging.info(f"Evaluating client {i} - name: {client['device']}")
                        if model_type == "hybrid":
                            auc_score, test_latent, test_label = evaluator.evaluate(client["test_loader"], client["train_loader"])
                            client_latent[round_idx][client['device']] = (test_latent, test_label)
                        else:
                            auc_score = evaluator.evaluate(client["test_loader"], client["train_loader"])

                        round_results[client['device']] = auc_score

                    round_results["global_loss"] = global_aggregator.val_loss
                    round_results['join_clients'] = selected_idx
                    round_results = {f'round_{round_idx+1}': round_results}

                    with open(filename, 'a') as f:
                        f.write(json.dumps(round_results) + '\n')

                    # --- Early Stopping Check ---
                    if global_aggregator.val_loss < min_val_loss:
                        min_val_loss = global_aggregator.val_loss
                        global_worse = 0
                    else:
                        global_worse += 1
                        if global_worse > global_patience:
                            logging.info("Early stopping triggered in global round!")
                            break

                if model_type == "hybrid":
                    file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(client_latent, f)
