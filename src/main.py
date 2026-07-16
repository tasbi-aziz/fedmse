"""
This is training endpoint updated with dual-pathway gateway routing.
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
from torch.utils.data import DataLoader, random_split, ConcatDataset
from Model import Shrink_Autoencoder
from Model import Autoencoder
from DataLoader import load_data
from DataLoader import IoTDataset
from DataLoader import IoTDataProccessor
from Trainer import ClientTrainer
from Trainer import GlobalAggregator
from Evaluator import Evaluator

# Import security buffer for Phase 2 holding
# Make sure src/Trainer/security_buffer.py exists with your SecurityBuffer implementation
from Trainer.security_buffer import SecurityBuffer  

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

num_participants = 0.5
epoch = 2
num_rounds = 5
lr_rate = 1e-5
shrink_lambda = 5
network_size = 2
data_seed = 1234

no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 2
batch_size = 32

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" 
dim_features = 115   # nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT' 

config_file = "/content/fedmse/Configuration/scen2-nba-iot-10clients.json"

# Phase 1 setup (e.g., first 2 rounds of training for quick testing out of 5 rounds)
prelim_rounds = 2 

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    random.seed(data_seed)
    np.random.seed(data_seed)
    try:
        logging.info("Loading configuration...")
        with open("/content/fedmse/Configuration/scen2-nba-iot-10clients.json", "r") as config_f:
            config = json.load(config_f)
    except Exception as e:
        logging.info("Failed to load configuration.")
        
    devices_list = random.sample(config['devices_list'], network_size)
    client_info = []

    for device in devices_list:
        logging.info("Creating metadata for client...")
        normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])

        abnormal_data_path = os.path.join(config['data_path'], device["normal_data_path"].replace("normal", "test_normal"))
        test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])

        logging.info("Loading data from {}...".format(device['name']))

        normal_data = load_data(normal_data_path)
        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        abnormal_data = load_data(abnormal_data_path)
        abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
        
        if new_device:
            new_normal_data = load_data(test_new_normal_data_path)
        
        device_name = device['name']
        print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")
        
        train_normal_size = int(0.4 * len(normal_data))
        valid_normal_size = int(0.1 * len(normal_data))
        dev_normal_size = int(0.4 * len(normal_data))
        test_normal_size = len(normal_data) - train_normal_size - valid_normal_size - dev_normal_size
        
        train_normal_data = normal_data[:train_normal_size]
        valid_normal_data = normal_data[train_normal_size:train_normal_size+valid_normal_size]
        dev_normal_data = normal_data[train_normal_size+valid_normal_size:train_normal_size+valid_normal_size+dev_normal_size]
        test_normal_data = normal_data[train_normal_size+valid_normal_size+dev_normal_size:]

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

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        
        # Save client latency simulations directly from config to track metrics
        client_info.append({
            "device": device['name'],
            "save_dir": "",
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
            "test_dataset": (processed_test_data, test_label),
            "dev_normal_dataset": dev_normal_data,
            "sim_train_time": device.get("simulated_training_time", 1.5),  # safe fallbacks if not present
            "sim_comm_time": device.get("simulated_comm_time", 0.5)
        })

    for update_type in ["avg", "fedprox", "mse_avg"]:
        for model_type in ["hybrid", "autoencoder"]:
            for run in range(num_runs):
                set_seeds(run*10000)
                for client in client_info:
                    client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", scen_name, model_type, update_type, client['device'])
                
                global_worse = 0
                min_val_loss = float("inf")

                directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                open(filename, 'w').close()
                
                # Model initializations
                if model_type == "hybrid":
                    global_model = Shrink_Autoencoder(input_dim=dim_features,
                                                      output_dim=dim_features,
                                                      shrink_lambda=shrink_lambda,
                                                      latent_dim=11,
                                                      hidden_neus=50)
                else:
                    global_model = Autoencoder(input_dim=dim_features,
                                               output_dim=dim_features,
                                               latent_dim=11,
                                               hidden_neus=50)
                    
                # Create Main model and separate verification holding buffer models
                global_model_buffer = copy.deepcopy(global_model)
                
                # Load aggregators
                global_aggregator = GlobalAggregator(global_model, update_type=update_type)
                buffer_aggregator = GlobalAggregator(global_model_buffer, update_type=update_type)
                
                # Initialize Temporal Security Buffer tracker
                sec_buffer_tracker = SecurityBuffer(
                    global_model=global_model,
                    window_size=5,
                    anomaly_threshold=1.5
                )

                # Set dev dataset for aggregators
                min_len = min([len(client['dev_normal_dataset']) for client in client_info])
                dev_dataset_sampled = []
                for client in client_info:
                    sample_data = client['dev_normal_dataset'].sample(n=min_len)
                    dev_dataset_sampled.append(sample_data)

                dev_dataset_sampled = np.concatenate(dev_dataset_sampled, axis=0)
                global_aggregator.create_dev_dataset({"dataset": dev_dataset_sampled})
                buffer_aggregator.create_dev_dataset({"dataset": dev_dataset_sampled})
                
                results = []
                client_latent = {}
                
                # Training Loop
                for round_idx in range(num_rounds):
                    if model_type == "hybrid":
                        client_latent[round_idx] = {}
                    
                    selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                    selected_clients = [client_info[i] for i in selected_idx]
                    
                    total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                    
                    # Store models mapped to paths
                    fast_path_weights = []
                    slow_path_weights = []
                    
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
                        
                        # Prepare weight structure for the standard aggregator list: (state_dict, total_samples, local_samples)
                        weight_entry = (raw_weights, total_training_samples, sample_count)
                        
                        # Apply Dual-Pathway Gatekeeper Routing logic
                        if round_idx < prelim_rounds:
                            # Phase 1: All updates process through the main lane
                            fast_path_weights.append(weight_entry)
                            
                            # Log metrics to calculate short-term threshold values
                            global_aggregator.record_phase1_metric(
                                round_idx=round_idx,
                                client_id=client['device'],
                                train_time=client['sim_train_time'],
                                comm_time=client['sim_comm_time'],
                                dataset_size=sample_count
                            )
                        else:
                            # Phase 2: Gateway Routing Check
                            is_fast = global_aggregator.evaluate_routing_lane(
                                train_time=client['sim_train_time'],
                                comm_time=client['sim_comm_time'],
                                dataset_size=sample_count
                            )
                            
                            if is_fast:
                                logging.info(f"⚡ {client['device']} -> FAST PATH")
                                fast_path_weights.append(weight_entry)
                            else:
                                logging.info(f"⏳ {client['device']} -> SLOW PATH (Held in Temporal Security Buffer)")
                                slow_path_weights.append(weight_entry)
                                
                                # Push slow client update into temporal tracking buffer
                                sec_buffer_tracker.add_to_buffer(client['device'], raw_weights)
                                
                        logging.info(f"Client {client['device']} training completed.")
                    
                    # --- Step 1: Execute aggregation for Fast lane ---
                    if fast_path_weights:
                        global_aggregator.update(local_models=fast_path_weights)
                    
                    # --- Step 2: Handle Phase-specific Operations ---
                    if round_idx < prelim_rounds:
                        # Phase 1: Calculate current short-term threshold
                        global_aggregator.calculate_round_st_threshold(round_idx)
                        
                        # Calculate long-term baseline threshold at the end of Phase 1
                        if round_idx == (prelim_rounds - 1):
                            global_aggregator.compute_final_lt_threshold()
                    else:
                        # Phase 2: Slow lane security checks and buffer updates
                        if slow_path_weights:
                            buffer_aggregator.update(local_models=slow_path_weights)
                        
                        # Extract verified clean parameter dictionaries from holding buffer
                        safe_extracted_weights = sec_buffer_tracker.extract_safe_updates()
                        
                        if safe_extracted_weights:
                            logging.info(f"Merging {len(safe_extracted_weights)} verified updates from slow lane back into global model...")
                            main_weights = global_aggregator.model.state_dict()
                            avg_extracted = {}
                            
                            for key in main_weights.keys():
                                target_device = global_aggregator.device
                                extracted_sum = sum(w[key].to(target_device) for w in safe_extracted_weights)
                                avg_extracted[key] = (main_weights[key] + extracted_sum) / (1 + len(safe_extracted_weights))
                                
                            global_aggregator.model.load_state_dict(avg_extracted)
                    
                    # Synchronize models
                    buffer_aggregator.model.load_state_dict(copy.deepcopy(global_aggregator.model.state_dict()))

                    logging.info(f"Round {round_idx+1}/{num_rounds} - Updated global model - Global loss: {global_aggregator.val_loss}")
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
                    
                    # Log results to output JSON
                    with open(filename, 'a') as f:
                        f.write(json.dumps(round_results) + '\n')
                    
                    # Check for global early stopping criteria
                    if global_aggregator.val_loss < min_val_loss:
                        min_val_loss = global_aggregator.val_loss
                        global_worse = 0
                    else:
                        global_worse += 1
                        if global_worse > global_patience:
                            logging.info("Early stopping triggered in global round!")
                            break
                
                # Save latent models for visualization analyses 
                if model_type == "hybrid":
                    file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(client_latent, f)
