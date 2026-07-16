"""
This is the global model update and routing gateway module.
Optimized to handle dual-model aggregation pathways and threshold calculations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from Utils import similarity_score
from DataLoader import IoTDataProccessor

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalAggregator(object):
    def __init__(self, model, update_type="avg", alpha=0.7, beta=0.3):
        """
        Initialize Global Aggregator with support for time-routing metrics.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_type = update_type
        self.model = model.to(self.device)
        
        # Gateway routing configuration
        self.alpha = alpha
        self.beta = beta
        
        # Buffers to track Phase 1 metrics (Rounds 1 to 20)
        # Structure: { round_num: { client_id: calculated_metric } }
        self.phase1_metrics = {}
        self.phase1_st_thresholds = []
        self.long_term_threshold = None

    def calculate_client_metric(self, train_time, comm_time, dataset_size):
        """
        Calculates the normalized routing metric for a client:
        T_client = alpha * (Train_Time + Comm_Time) + beta * Dataset_Size
        """
        latency = train_time + comm_time
        metric = (self.alpha * latency) + (self.beta * dataset_size)
        return metric

    def record_phase1_metric(self, round_idx, client_id, train_time, comm_time, dataset_size):
        """Stores calculated client metrics during Phase 1 rounds."""
        if round_idx not in self.phase1_metrics:
            self.phase1_metrics[round_idx] = {}
        
        metric = self.calculate_client_metric(train_time, comm_time, dataset_size)
        self.phase1_metrics[round_idx][client_id] = metric
        return metric

    def calculate_round_st_threshold(self, round_idx):
        """
        Calculates Short-Term Threshold (ST_Thrsh) for a specific round 
        using the Weighted Average Mean (WAM) of that round's client metrics.
        """
        round_data = self.phase1_metrics.get(round_idx, {})
        if not round_data:
            return 0.0
            
        metrics = list(round_data.values())
        # Using a standard mean as the base mathematical representation of WAM
        st_thrsh = float(np.mean(metrics))
        self.phase1_st_thresholds.append(st_thrsh)
        logging.info(f"Round {round_idx} ST_Thrsh (Short-Term Threshold) calculated: {st_thrsh:.4f}")
        return st_thrsh

    def compute_final_lt_threshold(self):
        """
        Computes the Long-Term Threshold (LT_Thrsh) at the end of Phase 1 
        using Exponential Weighted Moving Average (EWA) across all 20 rounds.
        """
        if not self.phase1_st_thresholds:
            logging.warning("No Short-Term Thresholds found. Defaulting LT_Thrsh to 0.")
            self.long_term_threshold = 0.0
            return 0.0

        # EWA factor (lambda weight) prioritizing more recent baseline rounds
        decay_factor = 0.9 
        running_ewa = self.phase1_st_thresholds[0]
        
        for st_val in self.phase1_st_thresholds[1:]:
            running_ewa = (decay_factor * st_val) + ((1 - decay_factor) * running_ewa)
            
        self.long_term_threshold = running_ewa
        logging.info(f"★★★ Phase 1 Complete! Long-Term Threshold (LT_Thrsh) set to: {self.long_term_threshold:.4f} ★★★")
        return self.long_term_threshold

    def evaluate_routing_lane(self, train_time, comm_time, dataset_size):
        """
        Gateway Routing Check:
        Returns True (Fast Path) if T_client < LT_Thrsh.
        Returns False (Slow Path) if T_client >= LT_Thrsh.
        """
        if self.long_term_threshold is None:
            # Fallback if Phase 2 is somehow called early
            return True
            
        t_client = self.calculate_client_metric(train_time, comm_time, dataset_size)
        is_fast = t_client < self.long_term_threshold
        return is_fast

    def create_dev_dataset(self, dataset):
        """
        Choose a development dataset for updating the global model (Fusion-based)
        """
        logging.info("Creating development dataset...")
        self.dev_dataset = dataset["dataset"]
        data_processor = IoTDataProccessor(scaler="standard")
        self.dev_dataset, self.dev_label = data_processor.fit_transform(self.dev_dataset)
        
        if self.update_type == "fusion_avg":
            self.dev_kde_scores = KernelDensity(kernel='gaussian', bandwidth="scott") \
                .fit(self.dev_dataset).score_samples(self.dev_dataset)
            
    def fusion_avg(self, local_models=None):
        """
        Perform fusion-based updating by calculating the average weights of local models.
        """
        logging.info("Fusion-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = similarity_score(self.dev_kde_scores, generated_data.cpu().numpy())
                weighted.append(1/sim_score)
                update_weights.append((local_model, 1/sim_score))
                
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
        self.model.load_state_dict(avg_weights)
        
    def fed_mse_avg(self, local_models=None):
        """
        Perform fusion-based updating using MSE loss on AE-based models.
        """
        logging.info("MSE-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model[0])
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)
                weighted.append(1/sim_score)
                update_weights.append((local_model[0], 1/sim_score))
                
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
                
        self.model.load_state_dict(avg_weights)
    
    def fed_avg(self, local_models=None):
        """
        Perform federated averaging to aggregate local model weights.
        """
        total_samples = sum(model[2] for model in local_models)
        avg_weights = {}

        for key in local_models[0][0].keys():
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        self.model.load_state_dict(avg_weights)
    
    def fedprox(self, local_models=None, mu=0.01):
        """
        Perform federated optimization using FedProx to aggregate local weights.
        """
        logging.info("FedProx updating...")
        total_samples = sum(model[2] for model in local_models)
        avg_weights = {}

        for key in local_models[0][0].keys():
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        self.model.load_state_dict(avg_weights)
    
    def update(self, local_models=None):
        """
        Update the global model weights.
        """
        if self.update_type == "avg":
            self.fed_avg(local_models)
        elif self.update_type == "fusion_avg":
            self.fusion_avg(local_models)
        elif self.update_type == "mse_avg":
            self.fed_mse_avg(local_models)
        elif self.update_type == "fedprox":
            self.fedprox(local_models, mu=0.01)
        
        self.model.eval()
        with torch.no_grad():
            _, _, val_loss = self.model(torch.Tensor(self.dev_dataset).to(self.device))
            self.val_loss = val_loss.item()
