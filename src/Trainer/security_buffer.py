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
        Initializes the security buffer.
        
        Args:
            global_model: The current active global PyTorch model.
            window_size (int): Maximum history of updates stored per client.
            similarity_threshold (float): Minimum cosine similarity required to be deemed normal (default: 0.7).
            trust_penalty (float): How much a client's trust drops on suspicious updates.
            trust_reward (float): How much a client's trust recovers on verified clean updates.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.trust_penalty = trust_penalty
        self.trust_reward = trust_reward
        
        # Historical registry of updates: { client_id: [state_dict_1, state_dict_2, ...] }
        self.buffer = {}
        
        # Running reputation system: { client_id: trust_score } (initialized at 1.0, bounded [0.0, 1.0])
        self.trust_scores = {}
        
    def add_to_buffer(self, client_id, local_model_state):
        """
        Saves a slow client's model weights and initializes their trust score if new.
        """
        if client_id not in self.buffer:
            self.buffer[client_id] = []
        
        # Initialize new clients with perfect trust (1.0)
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = 1.0
            
        # Store a deep copy of the weights
        self.buffer[client_id].append(copy.deepcopy(local_model_state))
        
        # Maintain sliding window size
        if len(self.buffer[client_id]) > self.window_size:
            self.buffer[client_id].pop(0)
            
        logging.info(f"[Security Buffer] Stored slow update for {client_id}. Trust Score: {self.trust_scores[client_id]:.2f}")

    def _flatten_parameters(self, state_dict):
        """
        Flattens all weight tensors into a single 1D vector for robust comparison.
        """
        tensors = []
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensors.append(value.flatten().float())
        return torch.cat(tensors)

    def _calculate_cosine_similarity(self, state_dict_a, state_dict_b):
        """
        Calculates the cosine similarity between two models.
        Values range from -1.0 (opposite directions) to 1.0 (perfect alignment).
        """
        vector_a = self._flatten_parameters(state_dict_a)
        vector_b = self._flatten_parameters(state_dict_b)
        
        norm_a = torch.norm(vector_a)
        norm_b = torch.norm(vector_b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        cosine_sim = torch.dot(vector_a, vector_b) / (norm_a * norm_b)
        return cosine_sim.item()

    def extract_safe_updates(self):
        """
        Evaluates buffered updates. Uses Cosine Similarity to verify alignment with 
        the global model, and dynamically adjusts client Trust Scores to filter stealthy attacks.
        
        Returns:
            list: A list of verified clean parameter dictionaries (state_dicts).
        """
        safe_updates = []
        global_state = self.global_model.state_dict()
        clients_to_clear = []
        
        for client_id, updates in list(self.buffer.items()):
            if not updates:
                continue
            
            # 1. Fetch latest slow update
            latest_update = updates[-1]
            
            # 2. Measure alignment direction with the active global model
            current_similarity = self._calculate_cosine_similarity(latest_update, global_state)
            
            # 3. Trajectory Stability check: Is similarity dropping compared to their previous rounds?
            sudden_drop = False
            if len(updates) > 1:
                prev_similarity = self._calculate_cosine_similarity(updates[-2], global_state)
                # If similarity drops significantly between rounds, flag as suspicious
                if current_similarity < (prev_similarity - 0.15):
                    sudden_drop = True
            
            # 4. Evaluate Trust Score Adjustment
            is_malicious = (current_similarity < self.similarity_threshold) or sudden_drop
            
            if is_malicious:
                # Penalize trust for suspicious or misaligned behavior
                self.trust_scores[client_id] = max(0.0, self.trust_scores[client_id] - self.trust_penalty)
                logging.warning(
                    f"⚠️ [Security Buffer] Anomaly from client {client_id}! "
                    f"Cosine Similarity: {current_similarity:.4f} (Drop: {sudden_drop}). "
                    f"Trust reduced to: {self.trust_scores[client_id]:.2f}"
                )
            else:
                # Reward trust for consistent positive behavior
                self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] + self.trust_reward)
                logging.info(
                    f"✅ [Security Buffer] Client {client_id} aligned nicely. "
                    f"Cosine Similarity: {current_similarity:.4f}. Trust: {self.trust_scores[client_id]:.2f}"
                )
                
            # 5. Gateway release decision
            # Update is only released if current behavior is good AND general trust is high (above 0.5)
            if not is_malicious and self.trust_scores[client_id] >= 0.5:
                logging.info(f"🔓 [Security Buffer] Releasing update for client {client_id} to global model.")
                safe_updates.append(latest_update)
                clients_to_clear.append(client_id)
            else:
                logging.warning(f"🔒 [Security Buffer] Client {client_id} remains quarantined (Insufficient trust/alignment).")
                
        # Clear successfully processed updates from buffer
        for client_id in clients_to_clear:
            self.buffer[client_id] = []
            
        return safe_updates
