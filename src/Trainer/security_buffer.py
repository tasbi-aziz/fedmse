"""
Security Gate & Time Buffer Module for FedMSE (Sync/Async Compatible).
Integrates Workload-Scaled Cosine Similarity, Sub-Sample Loss Variance,
and Dynamic Latency Thresholding alongside Time-Based Staging Buffers.
"""

import copy
import logging
import time
import numpy as np
import torch


class SecurityBuffer:
    def __init__(
        self, 
        global_model, 
        window_size=5, 
        time_buffer_seconds=30.0,
        base_similarity_threshold=0.65, 
        max_variance_threshold=2.0,
        trust_penalty=0.02, 
        trust_reward=0.05
    ):
        """
        Security Gate with Hybrid Async/Sync Support & Time Buffer.
        
        Args:
            global_model: Reference global model on the server.
            window_size: Max history rounds kept in quarantine.
            time_buffer_seconds: Time window for holding updates in Async mode.
            base_similarity_threshold: Base cosine similarity threshold (tau_0).
            max_variance_threshold: Limit for client 4-fold validation variance.
            trust_penalty: Trust score reduction on anomalous update.
            trust_reward: Trust score increase on aligned update.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.time_buffer_seconds = time_buffer_seconds
        self.base_similarity_threshold = base_similarity_threshold
        self.max_variance_threshold = max_variance_threshold
        self.trust_penalty = trust_penalty
        self.trust_reward = trust_reward
        
        # -------------------------------------------------------------
        # TIME BUFFER & STAGING POOL (Preserved for Async FL)
        # -------------------------------------------------------------
        # Staging buffer format: { client_id: { 'state': model_state, 'timestamp': float, ... } }
        self.staging_buffer = {}
        
        # Quarantine holding pool: { client_id: [state_dict_1, state_dict_2, ...] }
        self.quarantine_buffer = {}
        
        # Trajectory tracking & dynamic trust scores
        self.similarity_history = {}
        self.trust_scores = {}

    # =========================================================================
    # TIME BUFFER MANAGEMENT (ASYNC FL READY)
    # =========================================================================
    
    def add_to_time_buffer(self, client_id, local_model_state, dataset_size, val_loss_variance=0.0):
        """
        Holds update in Time Buffer for Async FL processing.
        """
        current_time = time.time()
        self.staging_buffer[client_id] = {
            'state': copy.deepcopy(local_model_state),
            'timestamp': current_time,
            'dataset_size': dataset_size,
            'val_loss_variance': val_loss_variance
        }
        logging.info(f"[Time Buffer] Update from Client {client_id} stored at {current_time:.2f}.")

    def flush_ready_time_buffer_updates(self):
        """
        Extracts updates that have waited beyond time_buffer_seconds.
        Used when Async FL mode is enabled.
        """
        current_time = time.time()
        ready_updates = {}
        expired_clients = []

        for client_id, data in self.staging_buffer.items():
            if (current_time - data['timestamp']) >= self.time_buffer_seconds:
                ready_updates[client_id] = data
                expired_clients.append(client_id)

        for client_id in expired_clients:
            del self.staging_buffer[client_id]

        return ready_updates

    # =========================================================================
    # 3-LAYER SECURITY GATE (COSINE SIM, VARIANCE, LATENCY)
    # =========================================================================

    def _flatten_state_dict(self, state_dict):
        """Flattens PyTorch state_dict parameters into a 1D vector on CPU."""
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                tensors.append(state_dict[key].detach().cpu().float().flatten())
        return torch.cat(tensors)

    def calculate_cosine_similarity(self, local_state, global_state):
        """Calculates cosine similarity between local weights and global weights on CPU."""
        vec_local = self._flatten_state_dict(local_state)
        vec_global = self._flatten_state_dict(global_state)
        
        norm_local = torch.norm(vec_local)
        norm_global = torch.norm(vec_global)
        
        if norm_local == 0 or norm_global == 0:
            return 0.0
            
        cosine_sim = torch.dot(vec_local, vec_global) / (norm_local * norm_global)
        return float(cosine_sim.item())

    def evaluate_and_route_update(
        self, 
        client_id, 
        local_model_state, 
        dataset_size, 
        arrival_time, 
        n_avg, 
        val_loss_variance=0.0
    ):
        """
        Primary Admission Gate:
        1. Cosine Similarity Check (Workload-scaled threshold).
        2. Sub-Sample Loss Variance Check (4-fold validation).
        3. Latency Check (Time window comparison).
        
        Returns:
            route (str): "DIRECT_PATH" or "QUARANTINE"
            current_sim (float): Measured Cosine Similarity
            tau_sim (float): Scaled Cosine Similarity Threshold
        """
        global_state = self.global_model.state_dict()
        
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = 1.0
            self.quarantine_buffer[client_id] = []
            self.similarity_history[client_id] = []

        # Step 1: Compute Cosine Similarity
        current_sim = self.calculate_cosine_similarity(local_model_state, global_state)

        # Step 2: Dynamic Threshold Scaling: tau_k = tau_0 / sqrt(n_k / n_avg)
        r_k = dataset_size / n_avg if n_avg > 0 else 1.0
        tau_sim = self.base_similarity_threshold / (r_k ** 0.5)
        tau_sim = max(0.60, min(0.95, tau_sim))

        # Step 3: 3-Layer Security Checks
        is_excessive_drift = current_sim < tau_sim
        is_unstable_variance = val_loss_variance > self.max_variance_threshold
        is_excessive_latency = arrival_time > self.time_buffer_seconds

        if is_excessive_drift or is_unstable_variance or is_excessive_latency:
            # Route to Quarantine Path
            self.trust_scores[client_id] = max(0.0, self.trust_scores[client_id] - self.trust_penalty)
            self.quarantine_buffer[client_id].append(copy.deepcopy(local_model_state))
            
            if len(self.quarantine_buffer[client_id]) > self.window_size:
                self.quarantine_buffer[client_id].pop(0)

            reason_str = f"Sim_Fail={is_excessive_drift}, Var_Fail={is_unstable_variance}, Latency_Fail={is_excessive_latency}"
            logging.warning(
                f"[Security Gate] Anomaly/Delay Detected for Client {client_id}! ({reason_str}) "
                f"Sim: {current_sim:.4f} (Tau: {tau_sim:.4f}), Var: {val_loss_variance:.4f}, Latency: {arrival_time:.1f}s. "
                f"Trust: {self.trust_scores[client_id]:.2f} -> Routed to QUARANTINE."
            )
            return "QUARANTINE", current_sim, tau_sim

        # Passed all checks -> Route to Direct Path
        self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] + self.trust_reward)
        self.similarity_history[client_id].append(current_sim)

        logging.info(
            f"[Security Gate] Client {client_id} CLEAN & FAST "
            f"(Sim: {current_sim:.4f} >= Tau: {tau_sim:.4f}, Var: {val_loss_variance:.4f}, Latency: {arrival_time:.1f}s) "
            f"-> Routed to DIRECT PATH."
        )
        return "DIRECT_PATH", current_sim, tau_sim

    def process_quarantine_validation(self, evaluator_fn, validation_loader):
        """
        Runs server-side SAE reconstruction validation on quarantined updates.
        """
        released_updates = []
        clients_to_clear = []

        for client_id, updates in list(self.quarantine_buffer.items()):
            if not updates:
                continue
            
            latest_update = updates[-1]
            mse_score = evaluator_fn(latest_update, validation_loader)
            
            if mse_score <= 0.05 and self.trust_scores[client_id] >= 0.4:
                logging.info(f"[Quarantine Verification] Client {client_id} passed MSE check ({mse_score:.4f}). Releasing.")
                released_updates.append(latest_update)
                clients_to_clear.append(client_id)
            else:
                logging.warning(f"[Quarantine Verification] Client {client_id} FAILED MSE check ({mse_score:.4f}). Dropping attack.")
                self.quarantine_buffer[client_id] = []

        for cid in clients_to_clear:
            self.quarantine_buffer[cid] = []

        return released_updates
