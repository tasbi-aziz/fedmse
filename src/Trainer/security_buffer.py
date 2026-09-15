"""
Security Buffer implementation for Federated Learning.
Handles 3-way routing: Direct Aggregation, Security Buffer Hold, and Malicious Update Rejection.
"""

import copy
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SecurityBuffer:
    def __init__(self, global_model, window_size=5, latency_threshold=20.0, alpha=0.2, beta=0.01):
        """
        security buffer initialization.
        - window_size: Max rounds to hold delayed updates.
        - latency_threshold: Max allowed latency for direct aggregation.
        - alpha, beta: Weight factors for score computation.
        """
        self.global_model = global_model
        self.window_size = window_size
        self.latency_threshold = latency_threshold
        self.alpha = alpha
        self.beta = beta
        
        # Buffer structure to hold updates across rounds
        self.buffer = []

    def compute_cosine_similarity(self, model_update, global_model):
        """Computes cosine similarity between local update vectors and global model weights."""
        flat_update = []
        flat_global = []

        for name, param in model_update.items():
            if name in global_model.state_dict():
                flat_update.append(param.view(-1).float())
                flat_global.append(global_model.state_dict()[name].view(-1).float())

        if not flat_update:
            return 0.0

        vec_update = torch.cat(flat_update)
        vec_global = torch.cat(flat_global)

        cos_sim = nn.functional.cosine_similarity(vec_update.unsqueeze(0), vec_global.unsqueeze(0))
        return cos_sim.item()

    def collect_current_round_updates(self, incoming_updates, global_model, val_loader, criterion, global_mse, device="cpu"):
        """
        Processes incoming client updates and routes them:
        1. Direct Accept: Low latency and good performance score.
        2. Buffer Hold: High latency but potential valid updates stored for next rounds.
        3. Reject: Suspicious updates (negative similarity or high loss variance).
        """
        ready_updates = []
        next_buffer = []

        # Step 1: Process old updates stored in buffer
        for buffered_item in self.buffer:
            buffered_item['age'] += 1
            if buffered_item['age'] <= self.window_size:
                # Re-evaluate buffered update with current global model
                cos_sim = self.compute_cosine_similarity(buffered_item['weights'], global_model)
                if cos_sim > 0.0:
                    ready_updates.append(buffered_item['weights'])
                    logging.info(f"[Security Buffer] Released buffered update from client {buffered_item['client_id']} (Age: {buffered_item['age']})")
                else:
                    next_buffer.append(buffered_item)
            else:
                logging.info(f"[Security Buffer] Dropped expired update from client {buffered_item['client_id']}")

        # Step 2: Process new incoming updates
        for update in incoming_updates:
            client_id = update['client_id']
            weights = update['weights']
            arrival_time = update['arrival_time']
            val_loss = update['val_loss']

            cos_sim = self.compute_cosine_similarity(weights, global_model)

            # Filtering Criteria
            if cos_sim < -0.2:
                # Reject negative correlation (potential poison attack)
                logging.warning(f"[Security Buffer] REJECTED client {client_id}: Negative similarity ({cos_sim:.4f})")
                continue

            if arrival_time <= self.latency_threshold:
                # Direct Path
                ready_updates.append(weights)
                logging.info(f"[Security Buffer] DIRECT ACCEPT client {client_id} (Latency: {arrival_time:.2f}s)")
            else:
                # Buffer Path (Late arrival)
                next_buffer.append({
                    'client_id': client_id,
                    'weights': copy.deepcopy(weights),
                    'val_loss': val_loss,
                    'age': 0
                })
                logging.info(f"[Security Buffer] BUFFERED client {client_id} (Latency: {arrival_time:.2f}s > Threshold)")

        # Update internal buffer status
        self.buffer = next_buffer
        return ready_updates
