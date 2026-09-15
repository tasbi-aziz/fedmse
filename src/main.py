"""
Main Execution Script for Federated Learning with FedOpt & Security Buffer Routing.
Includes complete data processing, anomaly detection evaluation (Precision, Recall, F1, AUC),
Security Buffer tracking, multi-run experiment execution, and JSON reporting.
"""

import os
import json
import argparse
import copy
import random
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset

from DataLoader.dataloader import load_data, IoTDataset, IoTDataProccessor
from Trainer import ClientTrainer, GlobalAggregator
from Trainer.security_buffer import SecurityBuffer
from Model import Shrink_Autoencoder, Autoencoder

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Hyperparameters & Configurations
num_participants = 1.0
epoch = 10
num_rounds = 10
lr_rate = 1e-4
shrink_dim = 16
threshold_val = 0.2
network_size = 10
data_seed = 1234
num_runs = 5
batch_size = 64
dim_features = 64

config_file = "/content/fedmse/Configuration/scen2-nba-iot-10clients.json"


def set_seeds(seed):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_global_mse(global_model, val_loader, device="cpu"):
    """Computes baseline Reconstruction Loss (MSE) of global model on validation dataset."""
    if global_model is None or val_loader is None:
        return 0.0
    global_model.eval()
    total_loss, total_samples = 0.0, 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            outputs = global_model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / max(total_samples, 1)


def compute_reconstruction_errors(model, data_loader, device="cpu"):
    """Calculates sample-wise MSE reconstruction errors."""
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            # Per-sample MSE
            loss = criterion(outputs, inputs).mean(dim=1)
            errors.extend(loss.cpu().numpy())

    return np.array(errors)


def evaluate_anomaly_detection(model, test_loader, device="cpu"):
    """
    Evaluates model performance on anomaly detection test set.
    Returns Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    model.eval()
    y_true = []
    reconstruction_errors = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].cpu().numpy()
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            loss = criterion(outputs, inputs).mean(dim=1).cpu().numpy()
            reconstruction_errors.extend(loss)
            y_true.extend(labels)

    y_true = np.array(y_true)
    reconstruction_errors = np.array(reconstruction_errors)

    # Dynamic Threshold Selection based on Normal Data Percentile
    normal_errors = reconstruction_errors[y_true == 0]
    if len(normal_errors) > 0:
        threshold = np.percentile(normal_errors, 95)
    else:
        threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)

    y_pred = (reconstruction_errors > threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, reconstruction_errors)
    except ValueError:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Evaluation Pipeline with FedOpt")
    parser.add_argument("--latency_threshold", type=float, default=20.0, help="Latency limit for direct aggregation")
    parser.add_argument("--update_type", type=str, default="fedopt", help="Aggregation type: fedopt or weighted")
    parser.add_argument("--server_lr", type=float, default=1.0, help="Server-side learning rate for FedOpt")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output metrics log directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(data_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Execution started on target device: {device}")

    # Load Configuration JSON
    with open(config_file, "r") as config_f:
        config = json.load(config_f)

    devices_list = random.sample(config['devices_list'], network_size)
    client_info = []

    logging.info("Initializing Data Processors and Partitioning Client Datasets...")

    # Load Client Data Pipelines
    for dev in devices_list:
        normal_data_path = os.path.join(config['data_path'], dev["normal_data_path"])
        abnormal_data_path = os.path.join(config['data_path'], dev["normal_data_path"].replace("normal", "test_normal"))

        normal_data = load_data(normal_data_path).sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(abnormal_data_path).sample(frac=1).reset_index(drop=True)

        train_normal_size = int(0.4 * len(normal_data))
        valid_normal_size = int(0.1 * len(normal_data))

        train_normal_data = normal_data[:train_normal_size]
        valid_normal_data = normal_data[train_normal_size:train_normal_size + valid_normal_size]
        test_normal_data = normal_data[train_normal_size + valid_normal_size:]

        data_processor = IoTDataProccessor(scaler="standard", use_log_transform=True, n_selected_features=dim_features)
        
        processed_train_data, train_label = data_processor.fit_transform(train_normal_data, abnormal_dataframe=abnormal_data)
        dim_features = processed_train_data.shape[1]

        processed_valid_data, valid_label = data_processor.transform(valid_normal_data)
        processed_test_normal, test_normal_label = data_processor.transform(test_normal_data)
        processed_test_abnormal, test_abnormal_label = data_processor.transform(abnormal_data)

        # Merge Test Sets (Normal + Abnormal)
        test_data_combined = np.vstack([processed_test_normal, processed_test_abnormal])
        test_label_combined = np.hstack([np.zeros(len(processed_test_normal)), np.ones(len(processed_test_abnormal))])

        train_dataset = IoTDataset(processed_train_data, train_label)
        valid_dataset = IoTDataset(processed_valid_data, valid_label)
        test_dataset = IoTDataset(test_data_combined, test_label_combined)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

        sim_train = dev.get("simulated_training_time", random.uniform(1.0, 5.0))
        sim_comm = dev.get("simulated_comm_time", random.uniform(0.2, 1.5))

        client_dir = os.path.join(args.output_dir, dev['name'])
        os.makedirs(client_dir, exist_ok=True)

        client_info.append({
            "device": dev['name'],
            "save_dir": client_dir,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
            "sim_train_time": sim_train,
            "sim_comm_time": sim_comm
        })

    # Combined Global Validation Dataset
    server_val_dataset = ConcatDataset([c['valid_loader'].dataset for c in client_info])
    server_val_loader = DataLoader(dataset=server_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Combined Global Test Dataset
    server_test_dataset = ConcatDataset([c['test_loader'].dataset for c in client_info])
    server_test_loader = DataLoader(dataset=server_test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    all_experiment_results = {}

    # Main Experiment Loop over Architectures
    for model_type in ["hybrid", "autoencoder"]:
        logging.info(f"\n================ STARTING EXPERIMENTS FOR MODEL TYPE: {model_type.upper()} ================\n")
        all_experiment_results[model_type] = []

        for run in range(num_runs):
            run_seed = (run + 1) * 10000
            set_seeds(run_seed)
            logging.info(f"--- Starting Execution Run {run + 1}/{num_runs} (Seed: {run_seed}) ---")

            # Model Initialization
            if model_type == "hybrid":
                global_model = Shrink_Autoencoder(input_dim=dim_features, shrink_dim=shrink_dim, threshold=threshold_val)
            else:
                global_model = Autoencoder(input_dim=dim_features, latent_dim=shrink_dim)

            global_model.to(device)

            # Initialize FedOpt Aggregator & Adaptive Security Buffer
            global_aggregator = GlobalAggregator(
                model=global_model, 
                update_type=args.update_type, 
                server_lr=args.server_lr
            )
            sec_buffer_tracker = SecurityBuffer(
                global_model=global_model,
                window_size=5,
                latency_threshold=args.latency_threshold,
                alpha=0.2,
                beta=0.01
            )

            run_round_history = []

            for round_idx in range(num_rounds):
                round_start_time = time.time()
                logging.info(f"[Run {run + 1} | Model {model_type}] --- Round {round_idx + 1}/{num_rounds} ---")

                # Baseline Server MSE
                global_mse = evaluate_global_mse(global_aggregator.model, server_val_loader, device=device)
                incoming_updates = []

                # Client Local Training Phase
                for client in client_info:
                    c_start = time.time()
                    device_trainer = ClientTrainer(
                        model=global_aggregator.model,
                        save_dir=client['save_dir'],
                        epoch=epoch,
                        lr_rate=lr_rate,
                        update_type=args.update_type
                    )

                    # Execute Training
                    device_trainer.run(client["train_loader"], client["valid_loader"])
                    compute_time = time.time() - c_start

                    client_val_loss = getattr(device_trainer, "val_loss", 1.0)
                    raw_weights = copy.deepcopy(device_trainer.model.state_dict())
                    
                    # Latency calculation (Compute + Simulated Network Delay)
                    total_arrival_latency = compute_time + client['sim_comm_time']

                    incoming_updates.append({
                        "client_id": client['device'],
                        "weights": raw_weights,
                        "arrival_time": total_arrival_latency,
                        "val_loss": client_val_loss,
                        "val_loss_variance": getattr(device_trainer, "val_loss_variance", 0.0)
                    })

                # Route updates via 3-Way Security Buffer
                ready_updates = sec_buffer_tracker.collect_current_round_updates(
                    incoming_updates=incoming_updates,
                    global_model=global_aggregator.model,
                    val_loader=server_val_loader,
                    criterion=criterion,
                    global_mse=global_mse,
                    device=device
                )

                # Server Aggregation via FedOpt / FedAdam
                global_aggregator.aggregate(client_updates=ready_updates)

                # Round Evaluation
                round_duration = time.time() - round_start_time
                post_eval_mse = evaluate_global_mse(global_aggregator.model, server_val_loader, device=device)
                
                # Global Test Anomaly Metrics
                global_metrics = evaluate_anomaly_detection(global_aggregator.model, server_test_loader, device=device)

                round_record = {
                    "round": round_idx + 1,
                    "global_val_mse": post_eval_mse,
                    "test_precision": global_metrics["precision"],
                    "test_recall": global_metrics["recall"],
                    "test_f1": global_metrics["f1_score"],
                    "test_auc": global_metrics["auc_roc"],
                    "round_duration": round_duration,
                    "accepted_updates": len(ready_updates)
                }

                run_round_history.append(round_record)

                logging.info(
                    f"[Round {round_idx + 1} Finished] Val MSE: {post_eval_mse:.6f} | "
                    f"F1: {global_metrics['f1_score']:.4f} | AUC: {global_metrics['auc_roc']:.4f} | "
                    f"Duration: {round_duration:.2f}s"
                )

            all_experiment_results[model_type].append(run_round_history)

            # Save Model Checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"{model_type}_run{run + 1}_checkpoint.pth")
            torch.save(global_aggregator.model.state_dict(), checkpoint_path)
            logging.info(f"Saved run checkpoint to: {checkpoint_path}")

    # Summarize Run Metric Statistics (Mean ± Std)
    summary_report = {}
    for m_type in all_experiment_results:
        final_f1_scores = [run_data[-1]["test_f1"] for run_data in all_experiment_results[m_type]]
        final_auc_scores = [run_data[-1]["test_auc"] for run_data in all_experiment_results[m_type]]
        final_mse_scores = [run_data[-1]["global_val_mse"] for run_data in all_experiment_results[m_type]]

        summary_report[m_type] = {
            "mean_f1": float(np.mean(final_f1_scores)),
            "std_f1": float(np.std(final_f1_scores)),
            "mean_auc": float(np.mean(final_auc_scores)),
            "std_auc": float(np.std(final_auc_scores)),
            "mean_mse": float(np.mean(final_mse_scores)),
            "std_mse": float(np.std(final_mse_scores))
        }

    # Save Results to JSON
    output_json_struct = {
        "summary": summary_report,
        "detailed_runs": all_experiment_results
    }

    results_json_path = os.path.join(args.output_dir, "experiment_execution_results.json")
    with open(results_json_path, "w") as f:
        json.dump(output_json_struct, f, indent=4)

    logging.info(f"\n================ PIPELINE EXECUTION COMPLETED ================")
    logging.info(f"Results Summary saved successfully to: {results_json_path}")
