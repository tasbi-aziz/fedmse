"""
Security Buffer & Quarantine Module for FedMSE.
Implements 3-Way Routing:
1. Direct Aggregation: Sim >= T and Arrival <= Latency (Fast & Clean)
2. Time Buffer: Sim >= T and Arrival > Latency (Slow & Clean, aggregated in next round with alpha)
3. Quarantine: Sim < T (Anomaly check via |MSE_i - MSE_g|, aggregated in next round with beta if passed)
"""

import copy
import logging
import torch
import torch.nn.functional as F
import numpy as np

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SecurityBuffer:
    def __init__(
        self, 
        latency_threshold=0.5, 
        similarity_threshold_T=0.8, 
        alpha=0.5, 
        beta=0.25, 
        mse_diff_threshold=0.05
    ):
        """
        :param latency_threshold: Fixed latency cutoff (e.g., 0.5s or 0.27s)
        :param similarity_threshold_T: Minimum Cosine Similarity threshold (T)
        :param alpha: Trust weight for Time Buffer updates in the next round
        :param beta: Trust weight for Quarantine passed updates in the next round
        :param mse_diff_threshold: Maximum allowed |MSE_i - MSE_g| in Quarantine
        """
        self.latency_threshold = latency_threshold
        self.T = similarity_threshold_T
        self.alpha = alpha
        self.beta = beta
        self.mse_diff_threshold = mse_diff_threshold
        
        # Staging Queues for Next Round Aggregation
        self.time_buffer_queue = []
        self.quarantine_pass_queue = []

    def _flatten_state_dict(self, state_dict):
        """Flattens PyTorch state dict into a 1D Tensor."""
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                tensors.append(state_dict[key].detach().cpu().float().flatten())
        return torch.cat(tensors)

    def calculate_cosine_similarity(self, local_state, global_state):
        """Calculates cosine similarity between local weights and global weights (W0)."""
        vec_local = self._flatten_state_dict(local_state)
        vec_global = self._flatten_state_dict(global_state)
        
        norm_local = torch.norm(vec_local)
        norm_global = torch.norm(vec_global)
        
        if norm_local == 0 or norm_global == 0:
            return 0.0
            
        cosine_sim = torch.dot(vec_local, vec_global) / (norm_local * norm_global)
        return float(cosine_sim.item())

    def evaluate_and_route_update(self, client_id, local_model_state, arrival_time, global_model_state):
        """
        Evaluates incoming client update based on Cosine Sim and Arrival Latency.
        """
        sim = self.calculate_cosine_similarity(local_model_state, global_model_state)
        update_obj = {
            "client_id": client_id,
            "weights": copy.deepcopy(local_model_state),
            "weight_factor": 1.0
        }

        # Rule 1: Sim >= T and Arrival time <= Latency -> Direct Aggregation (Fast & Clean)
        if sim >= self.T and arrival_time <= self.latency_threshold:
            update_obj["weight_factor"] = 1.0
            logging.info(f"[Security Gate] Client {client_id} -> DIRECT (Sim: {sim:.4f} >= {self.T}, Latency: {arrival_time:.2f}s <= {self.latency_threshold}s)")
            return "DIRECT", sim, update_obj

        # Rule 2: Sim >= T and Arrival time > Latency -> Time Buffer (Slow & Clean)
        elif sim >= self.T and arrival_time > self.latency_threshold:
            update_obj["weight_factor"] = self.alpha
            self.time_buffer_queue.append(update_obj)
            logging.info(f"[Security Gate] Client {client_id} -> TIME BUFFER (Sim: {sim:.4f} >= {self.T}, Latency: {arrival_time:.2f}s > {self.latency_threshold}s)")
            return "TIME_BUFFER", sim, update_obj

        # Rule 3: Sim < T -> Quarantine (Anomaly / Malicious)
        else:
            update_obj["weight_factor"] = self.beta
            logging.warning(f"[Security Gate] Client {client_id} -> QUARANTINE (Sim: {sim:.4f} < {self.T})")
            return "QUARANTINE", sim, update_obj

    def evaluate_quarantine_update(self, update_obj, global_model, val_loader, criterion, device="cpu"):
        """
        Quarantine Inspection:
        Calculates MSE(i) for local weights and MSE(g) for global weights on val_sample_dataset.
        Drops update if |MSE(i) - MSE(g)| > threshold, otherwise stores for next round aggregation.
        """
        client_id = update_obj["client_id"]
        weights = update_obj["weights"]

        # Temporary model for local weight evaluation
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(weights)
        
        local_model.to(device)
        global_model.to(device)
        local_model.eval()
        global_model.eval()

        mse_i_list = []
        mse_g_list = []

        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                
                out_i = local_model(batch_x)
                out_g = global_model(batch_x)

                mse_i = criterion(out_i, batch_x).item()
                mse_g = criterion(out_g, batch_x).item()

                mse_i_list.append(mse_i)
                mse_g_list.append(mse_g)

        avg_mse_i = np.mean(mse_i_list)
        avg_mse_g = np.mean(mse_g_list)
        mse_diff = abs(avg_mse_i - avg_mse_g)

        if mse_diff > self.mse_diff_threshold:
            logging.warning(
                f"[Quarantine Check] Client {client_id} REJECTED & DROPPED! "
                f"|MSE_i ({avg_mse_i:.4f}) - MSE_g ({avg_mse_g:.4f})| = {mse_diff:.4f} > {self.mse_diff_threshold}"
            )
            return False, avg_mse_i, avg_mse_g
        else:
            logging.info(
                f"[Quarantine Check] Client {client_id} PASSED! "
                f"|MSE_i ({avg_mse_i:.4f}) - MSE_g ({avg_mse_g:.4f})| = {mse_diff:.4f} <= {self.mse_diff_threshold}. Stash for next round with beta={self.beta}."
            )
            self.quarantine_pass_queue.append(update_obj)
            return True, avg_mse_i, avg_mse_g

    def collect_current_round_updates(self, incoming_updates, global_model, val_loader, criterion, device="cpu"):
        """
        Executes routing, quarantine inspection, and returns ready updates for current round aggregation.
        """
        current_round_pool = []

        # 1. Retrieve stored updates from previous round's Time Buffer and Quarantine Pass queues
        for item in self.time_buffer_queue:
            current_round_pool.append(item)
        for item in self.quarantine_pass_queue:
            current_round_pool.append(item)

        # Clear queues for next round populating
        self.time_buffer_queue.clear()
        self.quarantine_pass_queue.clear()

        # 2. Process incoming updates from current round
        global_state = global_model.state_dict()
        for update in incoming_updates:
            cid = update["client_id"]
            w = update["weights"]
            arr_time = update["arrival_time"]

            route, sim, update_obj = self.evaluate_and_route_update(cid, w, arr_time, global_state)

            if route == "DIRECT":
                current_round_pool.append(update_obj)
            elif route == "QUARANTINE":
                self.evaluate_quarantine_update(update_obj, global_model, val_loader, criterion, device)

        return current_round_pool
