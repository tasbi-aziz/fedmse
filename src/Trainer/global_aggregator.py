import copy
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GlobalAggregator:
    def __init__(
        self,
        model: Union[nn.Module, dict],
        update_type: str = "fedopt",
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3
    ):
        """
        Global Aggregator for FedMSE using adaptive server-side optimization (FedOpt / FedAdam / FedAvg).
        
        :param model: Global PyTorch Model instance or state_dict
        :param update_type: Aggregation strategy ('fedopt', 'fedadam', 'fedavg')
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

        # Server Optimizer States (First and Second Moments)
        self.m_t = {}
        self.v_t = {}
        self._init_server_moments()

    def _get_state_dict(self) -> dict:
        """Helper to extract model state dict."""
        if hasattr(self.model, "state_dict"):
            return self.model.state_dict()
        return self.model

    def _init_server_moments(self):
        """Initializes server momentum buffers m_t and v_t matching global model parameters."""
        state_dict = self._get_state_dict()
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor) and param.is_floating_point():
                self.m_t[key] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                self.v_t[key] = torch.zeros_like(param, dtype=torch.float32, device="cpu")

    def get_global_parameters(self) -> dict:
        """Returns current global model state dict."""
        return copy.deepcopy(self._get_state_dict())

    def aggregate(
        self,
        client_updates: List[Dict[str, Any]] = None,
        client_models: List[Any] = None,
        updates: List[Any] = None,
        **kwargs
    ) -> Union[nn.Module, dict]:
        """
        Aggregates client model updates into the global model using weighted pseudo-gradients.
        Weighted Factor W_i = dataset_size * weight_factor (Trust Routing Factor)
        """
        target_updates = client_updates if client_updates is not None else (client_models if client_models is not None else updates)

        if not target_updates:
            logging.warning("[GlobalAggregator] No client updates available for aggregation.")
            return self.model

        extracted_states = []
        combined_weights = []

        # 1. Unpack weights, dataset size (n_i), and trust route weight factors (w_i)
        for item in target_updates:
            if isinstance(item, dict) and "weights" in item:
                state = item["weights"]
                n_i = item.get("dataset_size", 1)
                w_i = item.get("weight_factor", 1.0)
            elif hasattr(item, "state_dict"):
                state = item.state_dict()
                n_i = getattr(item, "dataset_size", 1)
                w_i = getattr(item, "weight_factor", 1.0)
            elif isinstance(item, dict):
                state = item
                n_i = 1
                w_i = 1.0
            else:
                raise ValueError(f"Unsupported format in client_updates item: {type(item)}")

            extracted_states.append(state)
            combined_weights.append(float(n_i * w_i))

        num_clients = len(extracted_states)
        total_weight = sum(combined_weights)

        if total_weight <= 0:
            logging.error("[GlobalAggregator] Total weight factor is 0 or negative. Skipping aggregation.")
            return self.model

        current_global_state = self._get_state_dict()
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        updated_global_state = copy.deepcopy(current_global_state)

        # 2. Compute Weighted Pseudo-Gradients: Delta_avg = sum(W_i * (W_client - W_global)) / sum(W_i)
        pseudo_gradients = {}
        for key in current_global_state.keys():
            param = current_global_state[key]
            if isinstance(param, torch.Tensor) and param.is_floating_point():
                delta_sum = torch.zeros_like(param, dtype=torch.float32, device=device)
                for i in range(num_clients):
                    client_param = extracted_states[i][key].to(device).float()
                    delta_sum += (client_param - param.to(device).float()) * combined_weights[i]
                
                pseudo_gradients[key] = delta_sum / total_weight

        # 3. Apply Optimization Strategy (FedOpt / FedAdam vs Standard FedAvg)
        if self.update_type in ["fedopt", "fedadam"]:
            for key in current_global_state.keys():
                param = current_global_state[key]
                if isinstance(param, torch.Tensor) and param.is_floating_point():
                    grad = pseudo_gradients[key].to("cpu")

                    if key not in self.m_t:
                        self.m_t[key] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                        self.v_t[key] = torch.zeros_like(param, dtype=torch.float32, device="cpu")

                    # Update First and Second Moments
                    self.m_t[key] = self.beta1 * self.m_t[key] + (1.0 - self.beta1) * grad
                    self.v_t[key] = self.beta2 * self.v_t[key] + (1.0 - self.beta2) * torch.square(grad)

                    # Adaptive Step Update Rule (FedAdam)
                    denom = torch.sqrt(self.v_t[key]) + self.tau
                    step_update = (self.server_lr * (self.m_t[key] / denom)).to(device)

                    updated_global_state[key] = param.to(device) + step_update
                else:
                    updated_global_state[key] = extracted_states[0][key].to(device)
        else:
            # Fallback to Weighted FedAvg
            for key in current_global_state.keys():
                param = current_global_state[key]
                if isinstance(param, torch.Tensor) and param.is_floating_point():
                    updated_global_state[key] = param.to(device) + pseudo_gradients[key]
                else:
                    updated_global_state[key] = extracted_states[0][key].to(device)

        # 4. Load updated weights into Global Model
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(updated_global_state)
        else:
            self.model = updated_global_state

        logging.info(
            f"[GlobalAggregator] Aggregated {num_clients} client updates via {self.update_type.upper()} | "
            f"Total Effective Weight: {total_weight:.2f}"
        )
        return self.model
