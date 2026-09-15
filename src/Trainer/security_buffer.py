"""
Adaptive Client-Relative Security Buffer & Quarantine Module for FedMSE.
Evaluates local updates dynamically and handles 3-Way Adaptive Routing:
- DIRECT: High Trust, Low Latency (weight_factor = 1.0)
- TIME BUFFER: High Trust, High Latency (weight_factor = alpha)
- QUARANTINE: Low Trust -> Secondary Inspection -> Release (weight_factor = beta) or Drop.
"""

import copy
import logging
import math
import numpy as np
import torch
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SecurityBuffer:
    def __init__(
        self, 
        global_model=None, 
        window_size=5,
        latency_threshold=20.0, 
        warmup_rounds=3,
        k_factor=2.0,
        weights=(0.35, 0.35, 0.20, 0.10),
        alpha=0.2,
        beta=0.01,
        **kwargs
    ):
        self.global_model = global_model
        self.window_size = window_size
        self.latency_threshold = latency_threshold
        self.warmup_rounds = warmup_rounds
        self.k_factor = k_factor
        self.w_norm, self.w_loss, self.w_hist, self.w_lat = weights
        self.alpha = alpha
        self.beta = beta

        # Per-Client Historical Memory
        self.client_history = defaultdict(lambda: {
            "norm": [], 
            "loss_imp": [], 
            "latency": [], 
            "trust": []
        })

        # Staging Queues for Aggregation
        self.time_buffer_queue = []
        self.quarantine_queue = []
        self.quarantine_pass_queue = []

    def _flatten_state_dict(self, state_dict):
        """Flattens PyTorch state dict into a 1D Tensor."""
        tensors = []
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                tensors.append(state_dict[key].detach().cpu().float().flatten())
        return torch.cat(tensors)

    def _compute_update_norm(self, local_state, global_state):
        """Computes L2 Norm of local update delta relative to global model."""
        vec_local = self._flatten_state_dict(local_state)
        vec_global = self._flatten_state_dict(global_state)
        delta_vec = vec_local - vec_global
        return torch.norm(delta_vec).item()

    def calculate_trust_score(self, client_id, update_norm, loss_imp, arrival_time):
        """Calculates 4-Factor Adaptive Trust Score S_i for client update."""
        hist = self.client_history[client_id]

        norm_mean = np.mean(hist["norm"]) if len(hist["norm"]) > 0 else update_norm
        norm_std = np.std(hist["norm"]) if len(hist["norm"]) > 1 else 1.0
        loss_std = np.std(hist["loss_imp"]) if len(hist["loss_imp"]) > 1 else 1.0
        eps = 1e-8

        s_norm = math.exp(-update_norm / (norm_mean + eps))
        s_loss = 1.0 / (1.0 + math.exp(-loss_imp / (loss_std + eps)))
        s_hist = math.exp(-abs(update_norm - norm_mean) / (norm_std + eps))
        s_latency = max(0.0, 1.0 - (arrival_time / self.latency_threshold))

        trust_score = (
            self.w_norm * s_norm +
            self.w_loss * s_loss +
            self.w_hist * s_hist +
            self.w_lat * s_latency
        )
        return float(trust_score)

    def evaluate_and_route_update(
        self, 
        client_id, 
        local_model_state, 
        arrival_time=0.0, 
        val_loss=0.0,
        val_loss_variance=0.0,
        global_mse=0.0,
        global_model_state=None,
        **kwargs
    ):
        """Evaluates incoming update and routes to DIRECT, TIME BUFFER, or QUARANTINE."""
        ref_global_state = global_model_state if global_model_state is not None else self.global_model.state_dict()

        update_norm = self._compute_update_norm(local_model_state, ref_global_state)
        loss_imp = global_mse - val_loss
        loss_diff = abs(val_loss - global_mse)

        trust_score = self.calculate_trust_score(client_id, update_norm, loss_imp, arrival_time)

        update_obj = {
            "client_id": client_id,
            "weights": copy.deepcopy(local_model_state),
            "weight_factor": 1.0,
            "val_loss": val_loss,
            "val_loss_variance": val_loss_variance,
            "loss_diff": loss_diff,
            "update_norm": update_norm,
            "trust_score": trust_score
        }

        hist = self.client_history[client_id]
        rounds_seen = len(hist["trust"])

        # Phase 1: WARM-UP PHASE
        if rounds_seen < self.warmup_rounds:
            hist["norm"].append(update_norm)
            hist["loss_imp"].append(loss_imp)
            hist["latency"].append(arrival_time)
            hist["trust"].append(trust_score)

            if arrival_time <= self.latency_threshold:
                update_obj["weight_factor"] = 1.0
                route = "DIRECT"
            else:
                update_obj["weight_factor"] = self.alpha
                self.time_buffer_queue.append(update_obj)
                route = "TIME_BUFFER"

            logging.info(
                f"[Security Gate] Client {client_id} -> {route} (WARM-UP Round {rounds_seen+1}/{self.warmup_rounds}, "
                f"Trust: {trust_score:.4f}, Norm: {update_norm:.4f})"
            )
            return route, trust_score, 0.0, update_obj

        # Phase 2: ADAPTIVE DYNAMIC ROUTING
        mu_trust = np.mean(hist["trust"])
        sigma_trust = np.std(hist["trust"]) if len(hist["trust"]) > 1 else 0.05
        adaptive_threshold = max(0.1, mu_trust - (self.k_factor * sigma_trust))

        if trust_score >= adaptive_threshold:
            if arrival_time <= self.latency_threshold:
                update_obj["weight_factor"] = 1.0
                route = "DIRECT"
                logging.info(
                    f"[Security Gate] Client {client_id} -> DIRECT "
                    f"(Trust: {trust_score:.4f} >= {adaptive_threshold:.4f}, Latency: {arrival_time:.2f}s)"
                )
            else:
                update_obj["weight_factor"] = self.alpha
                self.time_buffer_queue.append(update_obj)
                route = "TIME_BUFFER"
                logging.info(
                    f"[Security Gate] Client {client_id} -> TIME BUFFER "
                    f"(Trust: {trust_score:.4f} >= {adaptive_threshold:.4f}, Latency: {arrival_time:.2f}s > {self.latency_threshold}s)"
                )
        else:
            update_obj["weight_factor"] = self.beta
            self.quarantine_queue.append(update_obj)
            route = "QUARANTINE"
            logging.warning(
                f"[Security Gate] Client {client_id} -> QUARANTINE "
                f"(Trust: {trust_score:.4f} < {adaptive_threshold:.4f})"
            )

        hist["norm"].append(update_norm)
        hist["loss_imp"].append(loss_imp)
        hist["latency"].append(arrival_time)
        hist["trust"].append(trust_score)

        if len(hist["trust"]) > self.window_size:
            hist["norm"].pop(0)
            hist["loss_imp"].pop(0)
            hist["latency"].pop(0)
            hist["trust"].pop(0)

        return route, trust_score, adaptive_threshold, update_obj

    def evaluate_quarantine_update(self, update_obj, global_model, val_loader, criterion, device="cpu"):
        """Inspects Quarantine items on server val set."""
        client_id = update_obj["client_id"]
        weights = update_obj["weights"]

        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(weights)
        local_model.to(device)
        global_model.to(device)
        local_model.eval()
        global_model.eval()

        mse_i_list, mse_g_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                out_i = local_model(batch_x)
                out_g = global_model(batch_x)

                if isinstance(out_i, (tuple, list)): out_i = out_i[0]
                if isinstance(out_g, (tuple, list)): out_g = out_g[0]

                mse_i_list.append(criterion(out_i, batch_x).item())
                mse_g_list.append(criterion(out_g, batch_x).item())

        avg_mse_i = np.mean(mse_i_list)
        avg_mse_g = np.mean(mse_g_list)
        dev_loss_imp = avg_mse_g - avg_mse_i

        if dev_loss_imp >= -0.05:
            logging.info(f"[Quarantine Check] Client {client_id} PASSED! Stashed with beta={self.beta}.")
            self.quarantine_pass_queue.append(update_obj)
            return True, avg_mse_i, avg_mse_g
        else:
            logging.warning(f"[Quarantine Check] Client {client_id} REJECTED & DROPPED!")
            return False, avg_mse_i, avg_mse_g

    def collect_current_round_updates(self, incoming_updates, global_model, val_loader, criterion, global_mse=0.0, device="cpu"):
        """Collects DIRECT updates, plus pending Time Buffer and Quarantine Pass updates from previous rounds."""
        current_round_pool = []

        # 1. Drain slow clean updates from previous round's staging queues
        for item in self.time_buffer_queue:
            current_round_pool.append(item)
        for item in self.quarantine_pass_queue:
            current_round_pool.append(item)

        self.time_buffer_queue.clear()
        self.quarantine_pass_queue.clear()

        # 2. Process incoming updates for current round
        for update in incoming_updates:
            cid = update["client_id"]
            w = update["weights"]
            arr_time = update.get("arrival_time", 0.0)
            val_loss = update.get("val_loss", 0.0)
            val_loss_var = update.get("val_loss_variance", 0.0)

            route, score, dynamic_thresh, update_obj = self.evaluate_and_route_update(
                client_id=cid, 
                local_model_state=w, 
                arrival_time=arr_time, 
                val_loss=val_loss,
                val_loss_variance=val_loss_var,
                global_mse=global_mse,
                global_model_state=global_model.state_dict()
            )

            if route == "DIRECT":
                current_round_pool.append(update_obj)
            elif route == "QUARANTINE":
                self.evaluate_quarantine_update(update_obj, global_model, val_loader, criterion, device)

        return current_round_pool
