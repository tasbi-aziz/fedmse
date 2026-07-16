"""
Temporal Security Buffer for holding, analyzing, and verifying slow client updates
before merging them into the main global model.
"""

import copy
import torch
import logging

class SecurityBuffer:
    def __init__(self, global_model, window_size=5, anomaly_threshold=1.5):
        """
        Initializes the security buffer.
        
        Args:
            global_model: The current active global PyTorch model.
            window_size (int): Maximum history of updates stored per client.
            anomaly_threshold (float): Factor multiplier for baseline anomaly checks.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        
        # Historical registry: { client_id: [state_dict_1, state_dict_2, ...] }
        self.buffer = {}
        
    def add_to_buffer(self, client_id, local_model_state):
        """
        Saves a slow client's model weights into their historical queue.
        """
        if client_id not in self.buffer:
            self.buffer[client_id] = []
            
        # Append copy of the weights to prevent references from changing during training
        self.buffer[client_id].append(copy.deepcopy(local_model_state))
        
        # Keep the history bounded to the window size
        if len(self.buffer[client_id]) > self.window_size:
            self.buffer[client_id].pop(0)
            
        logging.info(f"[Security Buffer] Stored slow update for {client_id}. Buffer size: {len(self.buffer[client_id])}/{self.window_size}")

    def _calculate_parameter_distance(self, state_dict_a, state_dict_b):
        """
        Calculates the Euclidean distance ($L_2$ norm) between two model parameter sets.
        $$d = \\sqrt{\\sum_{i} (\\theta_{a,i} - \\theta_{b,i})^2}$$
        """
        total_sq_dist = 0.0
        for key in state_dict_a.keys():
            # Skip non-parameter tracking metadata if any exist
            if not isinstance(state_dict_a[key], torch.Tensor):
                continue
            
            diff = state_dict_a[key].float() - state_dict_b[key].float()
            total_sq_dist += torch.sum(diff ** 2).item()
            
        return total_sq_dist ** 0.5

    def extract_safe_updates(self):
        """
        Evaluates buffered updates. Compares the deviation of slow updates 
        against the global model baseline. Only updates that exhibit stable parameter 
        trajectories are released.
        
        Returns:
            list: A list of verified clean parameter dictionaries (state_dicts).
        """
        safe_updates = []
        global_state = self.global_model.state_dict()
        
        # Temporary list to track which processed updates we can clear from the active buffer
        clients_to_clear = []
        
        for client_id, updates in list(self.buffer.items()):
            if not updates:
                continue
            
            # Use the latest update submitted by this slow client
            latest_update = updates[-1]
            
            # Calculate L2 distance to current global model parameters
            distance = self._calculate_parameter_distance(latest_update, global_state)
            
            # If historical updates exist, verify trajectory stability
            is_stable = True
            if len(updates) > 1:
                prev_distance = self._calculate_parameter_distance(updates[-2], global_state)
                # If distance jumps abnormally fast, flag as potentially malicious/poisoned
                if distance > (prev_distance * self.anomaly_threshold):
                    is_stable = False
                    logging.warning(f"⚠️ [Security Buffer] Anomaly detected for client {client_id}! Distance jumped from {prev_distance:.4f} to {distance:.4f}.")
            
            if is_stable:
                logging.info(f"✅ [Security Buffer] Client {client_id} passed verification checks (L2 distance: {distance:.4f}).")
                safe_updates.append(latest_update)
                clients_to_clear.append(client_id)
            else:
                logging.warning(f"❌ [Security Buffer] Client {client_id} failed verification and remains quarantined in buffer.")
                
        # Clear successfully integrated client records to prevent re-aggregating old data
        for client_id in clients_to_clear:
            self.buffer[client_id] = []
            
        return safe_updates
