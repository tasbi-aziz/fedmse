"""
Global Aggregator Module for FedMSE (FedOpt / FedAdam Enabled).
Performs server-side adaptive optimization (FedOpt/FedAdam) using weighted client pseudo-gradients:
- Direct Updates: weight_factor = 1.0
- Time Buffer Updates: weight_factor = alpha
- Quarantine Passed Updates: weight_factor = beta
"""

import copy
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GlobalAggregator:
    def __init__(self, model, update_type="fedopt", server_lr=1.0, beta1=0.9, beta2=0.999, tau=1e-3):
        """
        :param model: Global PyTorch Model instance
        :param update_type: Aggregation strategy ('fedopt', 'fedadam', 'fedavg', 'weighted')
        :param server_lr: Server-side learning rate (eta_g)
        :param beta1: First moment decay factor for FedOpt
        :param beta2: Second moment decay factor for FedOpt
        :param tau: Adaptivity parameter to prevent division by zero
        """
        self.model = model
        self.update_type = update_type.lower()
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        # Server Optimizer States (First and Second Moments for FedOpt/FedAdam)
        self.m_t = {}
        self.v_t = {}
        self._init_server_moments()

    def _init_server_moments(self):
        """Initializes server momentum buffers m_t and v_t matching model parameters."""
        state_dict = self.model.state_dict() if hasattr(self.model, "state_dict") else self.model
        for key, param in state_dict.items():
            if param.is_floating_point():
                self.m_t[key] = torch.zeros_like(param, dtype=torch.float32)
                self.v_t[key] = torch.zeros_like(param, dtype=torch.float32)

    def aggregate(self, client_updates=None, client_models=None, updates=None, **kwargs):
        """
        Aggregates client model updates into the global model using FedOpt/FedAvg with trust factors.
        """
        target_updates = client_updates if client_updates is not None else (client_models if client_models is not None else updates)

        if not target_updates:
            logging.warning("[GlobalAggregator] No client updates available for aggregation.")
            return self.model

        extracted_states = []
        weight_factors = []

        # 1. Unpack weights and trust factors
        for item in target_updates:
            if isinstance(item, dict) and "weights" in item:
                extracted_states.append(item["weights"])
                weight_factors.append(item.get("weight_factor", 1.0))
            elif hasattr(item, "state_dict"):
                extracted_states.append(item.state_dict())
                weight_factors.append(1.0)
            elif isinstance(item, dict):
                extracted_states.append(item)
                weight_factors.append(1.0)
            else:
                raise ValueError(f"Unsupported format in client_updates item: {type(item)}")

        num_clients = len(extracted_states)
        current_global_state = self.model.state_dict() if hasattr(self.model, "state_dict") else self.model
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")

        total_weight_factor = sum(weight_factors)
        if total_weight_factor == 0:
            logging.error("[GlobalAggregator] Total weight factor is 0. Skipping aggregation.")
            return self.model

        updated_global_state = copy.deepcopy(current_global_state)

        # 2. Compute Weighted Pseudo-Gradients: Delta_avg = sum(w_i * (W_i - W_global)) / sum(w_i)
        pseudo_gradients = {}
        for key in current_global_state.keys():
            if current_global_state[key].is_floating_point():
                delta_sum = torch.zeros_like(current_global_state[key], dtype=torch.float32, device=device)
                for i in range(num_clients):
                    client_param = extracted_states[i][key].to(device).float()
                    delta_sum += (client_param - current_global_state[key].to(device).float()) * weight_factors[i]
                
                pseudo_gradients[key] = delta_sum / total_weight_factor

        # 3. Apply Optimization Logic (FedOpt / FedAdam vs Weighted FedAvg)
        if self.update_type in ["fedopt", "fedadam"]:
            for key in current_global_state.keys():
                if current_global_state[key].is_floating_point():
                    grad = pseudo_gradients[key]
                    
                    # Update First and Second Moments
                    self.m_t[key] = self.m_t[key].to(device)
                    self.v_t[key] = self.v_t[key].to(device)

                    self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * grad
                    self.v_t[key] = self.beta2 * self.v_t[key] + (1 - self.beta2) * (grad ** 2)

                    # Adaptive Step Update (FedAdam Rule)
                    denom = torch.sqrt(self.v_t[key]) + self.tau
                    updated_global_state[key] = current_global_state[key].to(device) + self.server_lr * (self.m_t[key] / denom)
                else:
                    updated_global_state[key] = extracted_states[0][key].to(device)
        else:
            # Fallback to Weighted Averaging
            for key in current_global_state.keys():
                if current_global_state[key].is_floating_point():
                    updated_global_state[key] = current_global_state[key].to(device) + pseudo_gradients[key]
                else:
                    updated_global_state[key] = extracted_states[0][key].to(device)

        # 4. Load updated state into Global Model
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(updated_global_state)
        else:
            self.model = updated_global_state

        logging.info(
            f"[GlobalAggregator] Aggregated {num_clients} updates via {self.update_type.upper()} "
            f"(Total Weight Factor: {total_weight_factor:.2f})."
        )
        return self.model
