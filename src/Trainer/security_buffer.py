"""
Security Buffer & Quarantine Module for FedMSE (Updated - Loss & Variance Based Routing).
Implements 3-Way Routing:
1. Direct Aggregation: |MSE_i - MSE_g| <= T, Variance <= Max_Var, Arrival <= Latency (Fast & Clean, weight_factor = 1.0)
2. Time Buffer: |MSE_i - MSE_g| <= T, Variance <= Max_Var, Arrival > Latency (Slow & Clean, aggregated in next round with alpha)
3. Quarantine: |MSE_i - MSE_g| > T or Variance > Max_Var (Suspicious Update -> Secondary Validation on Server Dev Set)
"""

import copy
import logging
import torch
import numpy as np

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SecurityBuffer:
    def __init__(
        self, 
        global_model=None, 
        window_size=5,
        latency_threshold=20.0, 
        base_similarity_threshold=0.65,
        max_variance_threshold=0.05,
        alpha=0.2,
        beta=0.01,
        mse_diff_threshold=0.003,
        variance_weight=0.5,
        **kwargs
    ):
        self.global_model = global_model
        self.window_size = window_size
        self.latency_threshold = latency_threshold
        self.base_similarity_threshold = base_similarity_threshold
        self.max_variance_threshold = max_variance_threshold
        self.alpha = alpha
        self.beta = beta
        self.mse_diff_threshold = mse_diff_threshold
        self.variance_weight = variance_weight
        
        # Staging Queues for Aggregation
        self.time_buffer_queue = []
        self.quarantine_queue = []       # Staging queue for raw quarantine items
        self.quarantine_pass_queue = []  # Staging queue for items that passed quarantine validation

    def _flatten_state_dict(self, state_dict):
        """Flattens PyTorch state dict into a 1D Tensor."""
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                tensors.append(state_dict[key].detach().cpu().float().flatten())
        return torch.cat(tensors)

    def calculate_cosine_similarity(self, local_state, global_state):
        """Calculates cosine similarity between local weights and global weights (kept for compatibility)."""
        vec_local = self._flatten_state_dict(local_state)
        vec_global = self._flatten_state_dict(global_state)
        
        norm_local = torch.norm(vec_local)
        norm_global = torch.norm(vec_global)
        
        if norm_local == 0 or norm_global == 0:
            return 0.0
            
        cosine_sim = torch.dot(vec_local, vec_global) / (norm_local * norm_global)
        return float(cosine_sim.item())

    def calculate_client_score(self, val_loss: float, val_variance: float) -> float:
        """Client Stability Score: Loss + (variance_weight * Variance). Lower is better."""
        return val_loss + (self.variance_weight * val_variance)

    def evaluate_and_route_update(
        self, 
        client_id, 
        local_model_state, 
        arrival_time=0.0, 
        val_loss=0.0,
        val_loss_variance=0.0,
        global_mse=0.0,
        **kwargs
    ):
        """
        Evaluates incoming client update based on MSE Difference from Global Loss, 
        Arrival Latency, and Local Loss Variance.
        """
        loss_diff = abs(val_loss - global_mse)
        score = self.calculate_client_score(val_loss, val_loss_variance)

        update_obj = {
            "client_id": client_id,
            "weights": copy.deepcopy(local_model_state),
            "weight_factor": 1.0,
            "val_loss": val_loss,
            "val_loss_variance": val_loss_variance,
            "loss_diff": loss_diff,
            "score": score
        }

        # Rule 1: DIRECT AGGREGATION
        if (loss_diff <= self.mse_diff_threshold and 
            val_loss_variance <= self.max_variance_threshold and 
            arrival_time <= self.latency_threshold):
            
            update_obj["weight_factor"] = 1.0
            logging.info(
                f"[Security Gate] Client {client_id} -> DIRECT "
                f"(Diff: {loss_diff:.4f} <= {self.mse_diff_threshold}, Var: {val_loss_variance:.4f}, Latency: {arrival_time:.2f}s)"
            )
            return "DIRECT", loss_diff, update_obj

        # Rule 2: TIME BUFFER
        elif (loss_diff <= self.mse_diff_threshold and 
              val_loss_variance <= self.max_variance_threshold and 
              arrival_time > self.latency_threshold):
            
            update_obj["weight_factor"] = self.alpha
            self.time_buffer_queue.append(update_obj)
            logging.info(
                f"[Security Gate] Client {client_id} -> TIME BUFFER "
                f"(Diff: {loss_diff:.4f} <= {self.mse_diff_threshold}, Latency: {arrival_time:.2f}s > {self.latency_threshold}s)"
            )
            return "TIME_BUFFER", loss_diff, update_obj

        # Rule 3: QUARANTINE
        else:
            update_obj["weight_factor"] = self.beta
            self.quarantine_queue.append(update_obj)
            logging.warning(
                f"[Security Gate] Client {client_id} -> QUARANTINE "
                f"(Diff: {loss_diff:.4f} > {self.mse_diff_threshold} or High Var: {val_loss_variance:.4f})"
            )
            return "QUARANTINE", loss_diff, update_obj

    def evaluate_quarantine_update(self, update_obj, global_model, val_loader, criterion, device="cpu"):
        """
        Quarantine Inspection:
        Calculates MSE(i) for quarantined weights and MSE(g) for global weights on central dev set.
        Drops update if |MSE(i) - MSE(g)| > threshold, otherwise stashes for next round with beta weight factor.
        """
        client_id = update_obj["client_id"]
        weights = update_obj["weights"]

        # Temporary model for local weight evaluation on server validation loader
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(weights)
        
        local_model.to(device)
        global_model.to(device)
        local_model.eval()
        global_model.eval()

        mse_i_list = []
        mse_g_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                
                out_i = local_model(batch_x)
                out_g = global_model(batch_x)

                # Autoencoder output handling (if tuple returned)
                if isinstance(out_i, (tuple, list)):
                    out_i = out_i[0]
                if isinstance(out_g, (tuple, list)):
                    out_g = out_g[0]

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
                f"|MSE_i ({avg_mse_i:.4f}) - MSE_g ({avg_mse_g:.4f})| = {mse_diff:.4f} <= {self.mse_diff_threshold}. Stashed with beta={self.beta}."
            )
            self.quarantine_pass_queue.append(update_obj)
            return True, avg_mse_i, avg_mse_g

    def process_quarantine_validation(
        self, 
        global_model=None, 
        val_loader=None, 
        criterion=None, 
        device="cpu", 
        quarantine_list=None,
        validation_loader=None,
        **kwargs
    ):
        """Validates updates sitting in quarantine queue using server dev set."""
        if val_loader is None and validation_loader is not None:
            val_loader = validation_loader

        if global_model is None:
            global_model = self.global_model

        if criterion is None:
            criterion = torch.nn.MSELoss()

        released_updates = []
        targets = quarantine_list if quarantine_list is not None else self.quarantine_queue

        for update_obj in list(targets):
            passed, avg_mse_i, avg_mse_g = self.evaluate_quarantine_update(
                update_obj=update_obj,
                global_model=global_model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )
            if passed:
                released_updates.append(update_obj)

        if quarantine_list is None:
            self.quarantine_queue.clear()

        return released_updates

    def collect_current_round_updates(self, incoming_updates, global_model, val_loader, criterion, global_mse=0.0, device="cpu"):
        """
        Executes routing, quarantine inspection, and returns ready updates for current round aggregation.
        """
        current_round_pool = []

        # 1. Retrieve stored updates from previous round's Time Buffer and Quarantine Pass queues
        for item in self.time_buffer_queue:
            current_round_pool.append(item)
        for item in self.quarantine_pass_queue:
            current_round_pool.append(item)

        # Clear queues for current round populating
        self.time_buffer_queue.clear()
        self.quarantine_pass_queue.clear()

        # 2. Process incoming updates from current round
        for update in incoming_updates:
            cid = update["client_id"]
            w = update["weights"]
            arr_time = update.get("arrival_time", 0.0)
            val_loss = update.get("val_loss", 0.0)
            val_loss_var = update.get("val_loss_variance", 0.0)

            route, loss_diff, update_obj = self.evaluate_and_route_update(
                client_id=cid, 
                local_model_state=w, 
                arrival_time=arr_time, 
                val_loss=val_loss,
                val_loss_variance=val_loss_var,
                global_mse=global_mse
            )

            if route == "DIRECT":
                current_round_pool.append(update_obj)
            elif route == "QUARANTINE":
                self.evaluate_quarantine_update(update_obj, global_model, val_loader, criterion, device)

        return current_round_pool
