import copy
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SecurityBuffer:
    def __init__(
        self,
        global_model: nn.Module = None,
        window_size: int = 5,
        latency_threshold: float = 10.0,
        alpha: float = 0.2,
        beta: float = 0.01,
        cos_sim_threshold: float = -0.1
    ):
        """
        Security Buffer Initialization for Federated Learning.
        - window_size: Max rounds to hold delayed/straggler updates.
        - latency_threshold: Max allowed latency/execution time for direct aggregation.
        - alpha, beta: Weight factors for behavioral score computation.
        - cos_sim_threshold: Minimum cosine similarity threshold to reject malicious updates.
        """
        self.global_model = copy.deepcopy(global_model) if global_model is not None else None
        self.window_size = window_size
        self.latency_threshold = latency_threshold
        self.alpha = alpha
        self.beta = beta
        self.cos_sim_threshold = cos_sim_threshold
        
        # Buffer structure to hold delayed updates across rounds
        self.buffer: List[Dict[str, Any]] = []

    def set_global_model(self, global_model: nn.Module):
        """Updates the internal reference of the global model."""
        if global_model is not None:
            self.global_model = copy.deepcopy(global_model)

    def compute_cosine_similarity(self, model_update: dict, global_model: nn.Module) -> float:
        """
        Computes cosine similarity between local update vectors and global model weights.
        Ensures all tensors are moved to CPU to prevent CUDA/CPU device mismatch errors.
        """
        if global_model is None or not model_update:
            return 1.0

        flat_update = []
        flat_global = []

        global_dict = global_model.state_dict()

        for name, param in model_update.items():
            if name in global_dict:
                # Align both tensors on CPU to handle device mismatches
                flat_update.append(param.detach().cpu().view(-1).float())
                flat_global.append(global_dict[name].detach().cpu().view(-1).float())

        if not flat_update:
            return 0.0

        vec_update = torch.cat(flat_update)
        vec_global = torch.cat(flat_global)

        cos_sim = nn.functional.cosine_similarity(vec_update.unsqueeze(0), vec_global.unsqueeze(0))
        return cos_sim.item()

    def compute_behavioral_score(self, val_mse_list: list, cos_sim: float, global_mse: float = None, val_loss: float = 0.0) -> float:
        """
        Computes performance and behavioral consistency score using MSE metrics and Cosine Similarity.
        """
        if not val_mse_list:
            penalty = (self.beta * val_loss) if val_loss else 0.0
            return cos_sim - penalty

        mse_tensor = torch.tensor(val_mse_list, dtype=torch.float32)
        mean_mse = torch.mean(mse_tensor).item()
        var_mse = torch.var(mse_tensor, unbiased=False).item() if len(val_mse_list) > 1 else 0.0

        score = cos_sim - (self.alpha * var_mse) - (self.beta * mean_mse)
        return score

    def collect_current_round_updates(
        self, 
        incoming_updates: List[Dict[str, Any]], 
        global_model: nn.Module = None, 
        val_loader = None, 
        criterion = None, 
        global_mse: float = None, 
        device: str = "cpu",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Processes incoming client payloads and routes them via 3-way logic:
        1. Direct Accept: Low latency & passes security evaluation.
        2. Buffer Hold: High latency (straggler) stored for future rounds.
        3. Reject: Poisoned/malicious updates.
        """
        if global_model is not None:
            self.set_global_model(global_model)

        ready_updates = []
        next_buffer = []

        # -------------------------------------------------------------
        # STEP 1: Process and re-evaluate old updates stored in Buffer
        # -------------------------------------------------------------
        for buffered_item in self.buffer:
            buffered_item['age'] += 1
            client_id = buffered_item.get('client_id', 'unknown')

            if buffered_item['age'] <= self.window_size:
                cos_sim = self.compute_cosine_similarity(buffered_item['weights'], self.global_model)
                
                if cos_sim >= self.cos_sim_threshold and cos_sim > 0.0:
                    ready_updates.append(buffered_item)
                    logging.info(
                        f"[Security Buffer] RELEASED buffered update | Client: {client_id} | "
                        f"Age: {buffered_item['age']} | CosSim: {cos_sim:.4f}"
                    )
                else:
                    next_buffer.append(buffered_item)
                    logging.info(f"[Security Buffer] HOLDING buffered update | Client: {client_id} | Age: {buffered_item['age']}")
            else:
                logging.warning(f"[Security Buffer] DROPPED expired update | Client: {client_id} | Reached Max Age ({self.window_size})")

        # -------------------------------------------------------------
        # STEP 2: Process new incoming client updates
        # -------------------------------------------------------------
        for update in incoming_updates:
            client_id = update.get('client_id', -1)
            weights = update.get('weights', {})
            latency = update.get('arrival_time', update.get('train_time', 0.0))
            val_mse_list = update.get('val_mse_list', [])
            val_loss = update.get('val_loss', 0.0)

            # Compute Cosine Similarity (Device-Safe)
            cos_sim = self.compute_cosine_similarity(weights, self.global_model)
            score = self.compute_behavioral_score(val_mse_list, cos_sim, global_mse, val_loss)
            
            update['score'] = score
            update['cos_sim'] = cos_sim

            # Filter 1: Reject Poison / Malicious Updates
            if cos_sim < self.cos_sim_threshold:
                logging.warning(
                    f"[Security Buffer] REJECTED Client {client_id}: Malicious update detected "
                    f"(CosSim: {cos_sim:.4f} < Threshold: {self.cos_sim_threshold})"
                )
                continue

            # Filter 2: Route based on Latency / Execution Time
            if latency <= self.latency_threshold:
                ready_updates.append(update)
                logging.info(
                    f"[Security Buffer] DIRECT ACCEPT Client {client_id} | "
                    f"Latency: {latency:.2f}s | CosSim: {cos_sim:.4f} | Score: {score:.4f}"
                )
            else:
                buffered_payload = copy.deepcopy(update)
                buffered_payload['age'] = 0
                next_buffer.append(buffered_payload)
                logging.info(
                    f"[Security Buffer] BUFFERED Client {client_id} | "
                    f"Latency: {latency:.2f}s > Threshold ({self.latency_threshold}s) | Age: 0"
                )

        # Update internal state of buffer
        self.buffer = next_buffer
        logging.info(f"[Security Buffer] Summary -> Direct/Released Updates: {len(ready_updates)} | Active in Buffer: {len(self.buffer)}")

        return ready_updates

    def get_buffer_status(self) -> dict:
        """Returns diagnostic status of the security buffer."""
        return {
            "buffered_count": len(self.buffer),
            "buffered_clients": [item.get('client_id') for item in self.buffer],
            "window_size": self.window_size,
            "latency_threshold": self.latency_threshold
        }
