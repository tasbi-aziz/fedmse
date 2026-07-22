"""
Temporal Security Buffer utilizing Cosine Similarity and a running Trust Score 
to detect stealthy model poisoning attacks from slow clients.
"""

import copy
import torch
import logging

class SecurityBuffer:
    def __init__(self, global_model, window_size=5, similarity_threshold=0.7, trust_penalty=0.2, trust_reward=0.05):
        """
        Temporal Security Buffer utilizing Cosine Similarity on weight updates (\Delta \theta)
        and a running Trust Score to filter malicious updates.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.trust_penalty = trust_penalty
        self.trust_reward = trust_reward
        
        # Historical registry of updates: { client_id: [state_dict_1, state_dict_2, ...] }
        self.buffer = {}
        
        # Historical similarity registry to measure genuine trajectory drops: { client_id: [sim_1, sim_2, ...] }
        self.similarity_history = {}
        
        # Running reputation system: { client_id: trust_score }
        self.trust_scores = {}
        
    def add_to_buffer(self, client_id, local_model_state):
        if client_id not in self.buffer:
            self.buffer[client_id] = []
            self.similarity_history[client_id] = []
        
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = 1.0
            
        self.buffer[client_id].append(copy.deepcopy(local_model_state))
        
        if len(self.buffer[client_id]) > self.window_size:
            self.buffer[client_id].pop(0)
            self.similarity_history[client_id].pop(0)
            
        logging.info(f"[Security Buffer] Stored slow update for {client_id}. Trust Score: {self.trust_scores[client_id]:.2f}")

    def _flatten_diff(self, state_dict_a, state_dict_b):
        """
        Calculates and flattens weight difference (Delta = Model_A - Model_B) into a 1D vector.
        """
        tensors = []
        for key in state_dict_a.keys():
            if isinstance(state_dict_a[key], torch.Tensor) and key in state_dict_b:
                diff = state_dict_a[key].float() - state_dict_b[key].float()
                tensors.append(diff.flatten())
        return torch.cat(tensors)

    def _calculate_cosine_similarity(self, client_state, global_state):
        """
        Calculates cosine similarity based on weight updates relative to global state.
        """
        vector_a = self._flatten_diff(client_state, global_state)
        
        norm_a = torch.norm(vector_a)
        
        # Case: Client made 0 local changes (Idle / Zero gradient)
        if norm_a == 0:
            return 1.0, True  # (similarity, is_idle)
            
        # Compare client trajectory against current global direction
        global_vec = torch.cat([v.flatten().float() for k, v in global_state.items() if isinstance(v, torch.Tensor)])
        norm_g = torch.norm(global_vec)
        
        if norm_g == 0:
            return 0.0, False
            
        cosine_sim = torch.dot(vector_a, global_vec) / (norm_a * norm_g)
        return float(cosine_sim.item()), False

    def extract_safe_updates(self):
        safe_updates = []
        global_state = self.global_model.state_dict()
        clients_to_clear = []
        
        for client_id, updates in list(self.buffer.items()):
            if not updates:
                continue
            
            # 1. Fetch latest update
            latest_update = updates[-1]
            
            # 2. Measure cosine similarity
            current_similarity, is_idle = self._calculate_cosine_similarity(latest_update, global_state)
            
            # 3. Check trajectory drop safely using stored past values
            sudden_drop = False
            history = self.similarity_history[client_id]
            if len(history) > 0:
                prev_sim = history[-1]
                if current_similarity < (prev_sim - 0.25):  # 0.25 drop tolerance
                    sudden_drop = True
            
            # Store current valid similarity to historical tracker
            self.similarity_history[client_id].append(current_similarity)
            
            # 4. Evaluate Anomaly Status
            is_malicious = (current_similarity < self.similarity_threshold) or sudden_drop or is_idle
            
            if is_malicious:
                self.trust_scores[client_id] = max(0.0, self.trust_scores[client_id] - self.trust_penalty)
                logging.warning(
                    f" [Security Buffer] Anomaly from client {client_id}! "
                    f"Similarity: {current_similarity:.4f} (Drop: {sudden_drop}, Idle: {is_idle}). "
                    f"Trust reduced to: {self.trust_scores[client_id]:.2f}"
                )
            else:
                self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] + self.trust_reward)
                logging.info(
                    f" [Security Buffer] Client {client_id} aligned nicely. "
                    f"Similarity: {current_similarity:.4f}. Trust: {self.trust_scores[client_id]:.2f}"
                )
                
            # 5. Gateway release decision
            if not is_malicious and self.trust_scores[client_id] >= 0.5:
                logging.info(f" [Security Buffer] Releasing update for client {client_id} to global model.")
                safe_updates.append(latest_update)
                clients_to_clear.append(client_id)
            else:
                logging.warning(f" [Security Buffer] Client {client_id} remains quarantined.")
                
        for client_id in clients_to_clear:
            self.buffer[client_id] = []
            
        return safe_updates
