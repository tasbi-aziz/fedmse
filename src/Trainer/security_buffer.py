"""
Dataset-Aware Security Buffer for FedMSE.
Integrates Workload-Scaled Cosine Similarity (Security Gate) 
and Robust Dynamic Latency Thresholding (Synchronization Gate).
"""

import copy
import logging
import numpy as np
import torch

class SecurityBuffer:
    def __init__(
        self, 
        global_model, 
        window_size=5, 
        base_similarity_threshold=0.85, 
        latency_threshold=10.0,
        trust_penalty=0.2, 
        trust_reward=0.05
    ):
        """
        Multi-Signal Security Buffer.
        
        Args:
            global_model: Main model reference on the server.
            window_size: Max history rounds kept in quarantine.
            base_similarity_threshold: Base similarity threshold (tau_0).
            latency_threshold: Initial max seconds allowed for Direct Path.
            trust_penalty: Trust score reduction on anomalous update.
            trust_reward: Trust score increase on aligned update.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.base_similarity_threshold = base_similarity_threshold
        self.latency_threshold = latency_threshold
        self.trust_penalty = trust_penalty
        self.trust_reward = trust_reward
        
        # Quarantine holding pool: { client_id: [state_dict_1, state_dict_2, ...] }
        self.quarantine_buffer = {}
        
        # Stored similarity history for trajectory tracking: { client_id: [sim_1, ...] }
        self.similarity_history = {}
        
        # Dynamic trust scores: { client_id: score }
        self.trust_scores = {}

    def update_dynamic_latency_threshold(self, arrival_times, k=1.5):
        """
        Calculates dynamic latency threshold using: Median(T) + K * IQR(T).
        
        Args:
            arrival_times (list or np.ndarray): Arrival times of clients in current round.
            k (float): Outlier sensitivity multiplier (typically 1.5 to 2.0).
        """
        if not arrival_times or len(arrival_times) == 0:
            return self.latency_threshold

        times = np.array(arrival_times)
        median_t = np.median(times)
        q75, q25 = np.percentile(times, [75, 25])
        iqr_t = q75 - q25

        # Dynamic formula calculation
        calculated_threshold = median_t + (k * iqr_t)
        
        # Keep threshold at a safe positive non-zero floor
        self.latency_threshold = max(1.0, float(calculated_threshold))
        
        logging.info(
            f"[Synchronization Gate] Dynamic Latency Threshold updated to: "
            f"{self.latency_threshold:.2f}s (Median={median_t:.2f}s, IQR={iqr_t:.2f}s, k={k})"
        )
        return self.latency_threshold

    def _flatten_state_dict(self, state_dict):
        """
        Flattens PyTorch state_dict parameters into a 1D vector on CPU 
        to guarantee uniform device placement during vector operations.
        """
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                # Detach and force movement to CPU
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

    def evaluate_and_route_update(self, client_id, local_model_state, dataset_size, arrival_time, n_avg):
        """
        Primary Admission Gate.
        
        1. Scales Cosine Similarity threshold by workload ratio (n_k / n_avg).
        2. Phase 1 (Security): Checks weight drift against dynamic threshold.
        3. Phase 2 (Synchronization): Bypasses Quarantine for clean updates and routes by latency.
        """
        global_state = self.global_model.state_dict()
        
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = 1.0
            self.quarantine_buffer[client_id] = []
            self.similarity_history[client_id] = []

        # -------------------------------------------------------------
        # STEP 1: Compute Cosine Similarity
        # -------------------------------------------------------------
        current_sim = self.calculate_cosine_similarity(local_model_state, global_state)

        # -------------------------------------------------------------
        # STEP 2: Workload-Aware Similarity Threshold Scaling
        # Dynamic Formula: tau_k = tau_0 / sqrt(n_k / n_avg)
        # -------------------------------------------------------------
        r_k = dataset_size / n_avg if n_avg > 0 else 1.0
        tau_sim = self.base_similarity_threshold / (r_k ** 0.5)
        tau_sim = max(0.60, min(0.95, tau_sim))

        # -------------------------------------------------------------
        # STEP 3: PHASE 1 — Security Gate (Drift Evaluation)
        # -------------------------------------------------------------
        is_excessive_drift = current_sim < tau_sim

        if is_excessive_drift:
            self.trust_scores[client_id] = max(0.0, self.trust_scores[client_id] - self.trust_penalty)
            self.quarantine_buffer[client_id].append(copy.deepcopy(local_model_state))
            
            if len(self.quarantine_buffer[client_id]) > self.window_size:
                self.quarantine_buffer[client_id].pop(0)

            logging.warning(
                f"[Security Gate] Excessive Drift Detected for Client {client_id}! "
                f"Sim: {current_sim:.4f} < Tau: {tau_sim:.4f} (Workload ratio: {r_k:.2f}). "
                f"Trust reduced to: {self.trust_scores[client_id]:.2f} -> Sent to QUARANTINE."
            )
            return "QUARANTINE", current_sim, tau_sim

        # Reward clean client
        self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] + self.trust_reward)
        self.similarity_history[client_id].append(current_sim)

        # -------------------------------------------------------------
        # STEP 4: PHASE 2 — Synchronization Gate (Latency Routing)
        # -------------------------------------------------------------
        if arrival_time <= self.latency_threshold:
            logging.info(
                f"[Security Gate] Client {client_id} CLEAN & FAST "
                f"(Sim: {current_sim:.4f} >= Tau: {tau_sim:.4f}, Delay: {arrival_time:.1f}s <= Threshold: {self.latency_threshold:.1f}s) "
                f"-> Routed to DIRECT PATH."
            )
            return "DIRECT_PATH", current_sim, tau_sim
        else:
            logging.info(
                f"[Security Gate] Client {client_id} CLEAN but SLOW "
                f"(Sim: {current_sim:.4f} >= Tau: {tau_sim:.4f}, Delay: {arrival_time:.1f}s > Threshold: {self.latency_threshold:.1f}s) "
                f"-> Routed to TIME BUFFER."
            )
            return "TIME_BUFFER", current_sim, tau_sim

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
