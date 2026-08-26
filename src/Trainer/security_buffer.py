"""
Dataset-Aware Security Buffer for FedMSE.
Integrates Workload-Scaled Cosine Similarity (Security Gate) 
and Arrival Time Routing (Synchronization Gate).
"""

import copy
import logging
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
            latency_threshold: Max seconds allowed for Direct Path.
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

    def _flatten_state_dict(self, state_dict):
        """Flattens PyTorch state_dict parameters into a 1D vector."""
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                tensors.append(state_dict[key].float().flatten())
        return torch.cat(tensors)

    def calculate_cosine_similarity(self, local_state, global_state):
        """Calculates cosine similarity between local weights and global weights."""
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
        
        Returns:
            route_status (str): 'DIRECT_PATH', 'TIME_BUFFER', or 'QUARANTINE'
            current_sim (float): Measured Cosine Similarity
            tau_sim (float): Dynamic threshold calculated for this client
        """
        global_state = self.global_model.state_dict()
        
        # Initialize client registries if first interaction
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = 1.0
            self.quarantine_buffer[client_id] = []
            self.similarity_history[client_id] = []

        # -------------------------------------------------------------
        # STEP 1: Compute Cosine Similarity
        # -------------------------------------------------------------
        current_sim = self.calculate_cosine_similarity(local_model_state, global_state)

        # -------------------------------------------------------------
        # STEP 2: Workload-Aware Threshold Scaling
        # Dynamic Formula: tau_k = tau_0 / sqrt(n_k / n_avg)
        # Larger datasets naturally shift weights more, lowering the required similarity bound.
        # -------------------------------------------------------------
        r_k = dataset_size / n_avg if n_avg > 0 else 1.0
        tau_sim = self.base_similarity_threshold / (r_k ** 0.5)
        
        # Keep threshold within realistic boundary limits [0.60, 0.95]
        tau_sim = max(0.60, min(0.95, tau_sim))

        # -------------------------------------------------------------
        # STEP 3: PHASE 1 — Security Gate (Drift Evaluation)
        # -------------------------------------------------------------
        is_excessive_drift = current_sim < tau_sim

        if is_excessive_drift:
            # Penalize trust score and store in Quarantine Buffer
            self.trust_scores[client_id] = max(0.0, self.trust_scores[client_id] - self.trust_penalty)
            
            # Store copy in quarantine pool
            self.quarantine_buffer[client_id].append(copy.deepcopy(local_model_state))
            if len(self.quarantine_buffer[client_id]) > self.window_size:
                self.quarantine_buffer[client_id].pop(0)

            logging.warning(
                f"[Security Gate] Excessive Drift Detected for Client {client_id}! "
                f"Sim: {current_sim:.4f} < Tau: {tau_sim:.4f} (Workload ratio: {r_k:.2f}). "
                f"Trust reduced to: {self.trust_scores[client_id]:.2f} -> Sent to QUARANTINE."
            )
            return "QUARANTINE", current_sim, tau_sim

        # Update trust score for clean update
        self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] + self.trust_reward)
        self.similarity_history[client_id].append(current_sim)

        # -------------------------------------------------------------
        # STEP 4: PHASE 2 — Synchronization Gate (Latency Routing)
        # Clean updates bypass Quarantine completely!
        # -------------------------------------------------------------
        if arrival_time <= self.latency_threshold:
            logging.info(
                f"[Security Gate] Client {client_id} CLEAN & FAST "
                f"(Sim: {current_sim:.4f} >= Tau: {tau_sim:.4f}, Delay: {arrival_time:.1f}s) "
                f"-> Routed to DIRECT PATH."
            )
            return "DIRECT_PATH", current_sim, tau_sim
        else:
            logging.info(
                f"[Security Gate] Client {client_id} CLEAN but SLOW "
                f"(Sim: {current_sim:.4f} >= Tau: {tau_sim:.4f}, Delay: {arrival_time:.1f}s) "
                f"-> Routed to TIME BUFFER."
            )
            return "TIME_BUFFER", current_sim, tau_sim

    def process_quarantine_validation(self, evaluator_fn, validation_loader):
        """
        Runs server-side SAE reconstruction validation on quarantined updates.
        Clears malicious updates and returns verified updates for aggregation.
        """
        released_updates = []
        clients_to_clear = []

        for client_id, updates in list(self.quarantine_buffer.items()):
            if not updates:
                continue
            
            latest_update = updates[-1]
            
            # Run server-side reconstruction validation score
            mse_score = evaluator_fn(latest_update, validation_loader)
            
            # If MSE error is within reasonable reconstruction limits (e.g., < 0.05)
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
