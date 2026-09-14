"""
Global Aggregator Module for FedMSE.
Performs weighted parameter aggregation based on Security Gate trust factors:
- Direct Updates: weight_factor = 1.0
- Time Buffer Updates: weight_factor = alpha
- Quarantine Passed Updates: weight_factor = beta
"""

import copy
import logging
import torch

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GlobalAggregator:
    def __init__(self, model, update_type="weighted"):
        """
        :param model: Global PyTorch Model instance
        :param update_type: Aggregation strategy ('weighted', 'avg', or 'mse_avg')
        """
        self.model = model
        self.update_type = update_type

    def aggregate(self, client_updates=None, client_losses=None, client_models=None, updates=None, **kwargs):
        """
        Aggregates client model updates into the global model.
        Supports flexible keyword arguments ('client_updates', 'client_models', or 'updates').
        
        :param client_updates: List of client objects from SecurityBuffer 
                               [{"client_id": ..., "weights": state_dict, "weight_factor": factor}, ...]
                               or list of raw PyTorch models/state_dicts.
        :param client_losses: Optional list of validation losses for inverse-loss weighting.
        :param client_models: Alias for client_updates (for compatibility with main.py calls).
        :param updates: Alias for client_updates.
        """
        # Resolve inputs from all possible keyword arguments
        target_updates = client_updates
        if target_updates is None:
            target_updates = client_models
        if target_updates is None:
            target_updates = updates

        if not target_updates:
            logging.warning("[GlobalAggregator] No client updates available for aggregation.")
            return self.model

        extracted_states = []
        weight_factors = []

        # 1. Unpack weights and trust factors (alpha, beta, 1.0)
        for item in target_updates:
            if isinstance(item, dict) and "weights" in item:
                # Payload format from SecurityBuffer
                extracted_states.append(item["weights"])
                weight_factors.append(item.get("weight_factor", 1.0))
            elif hasattr(item, "state_dict"):
                # PyTorch Model instance
                extracted_states.append(item.state_dict())
                weight_factors.append(1.0)
            elif isinstance(item, dict):
                # Plain state_dict
                extracted_states.append(item)
                weight_factors.append(1.0)
            else:
                raise ValueError(f"Unsupported format in client_updates item: {type(item)}")

        num_clients = len(extracted_states)
        current_global_state = self.model.state_dict() if hasattr(self.model, "state_dict") else self.model
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")

        # 2. Weighted Aggregation Logic: sum(W_i * Factor_i) / sum(Factor_i)
        updated_global_state = copy.deepcopy(current_global_state)
        total_weight_factor = sum(weight_factors)

        if total_weight_factor == 0:
            logging.error("[GlobalAggregator] Total weight factor is 0. Skipping aggregation.")
            return self.model

        for key in current_global_state.keys():
            if current_global_state[key].is_floating_point():
                # Create zero accumulator on the model's device
                param_sum = torch.zeros_like(current_global_state[key], dtype=torch.float32, device=device)
                
                for i in range(num_clients):
                    client_param = extracted_states[i][key].to(device).float()
                    param_sum += client_param * weight_factors[i]
                
                # Dynamic normalization by sum of weight factors
                updated_global_state[key] = param_sum / total_weight_factor
            else:
                # Non-floating point parameters (e.g. step counters) assigned from first update
                updated_global_state[key] = extracted_states[0][key].to(device)

        # 3. Load updated state into Global Model
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(updated_global_state)
        else:
            self.model = updated_global_state

        logging.info(
            f"[GlobalAggregator] Aggregated {num_clients} updates into Global Model "
            f"(Total Trust Weight Factor: {total_weight_factor:.2f})."
        )
        return self.model
