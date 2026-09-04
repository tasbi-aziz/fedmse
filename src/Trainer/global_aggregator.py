import copy
import torch


class GlobalAggregator:
    def __init__(self, model, update_type="avg"):
        self.model = model
        self.update_type = update_type
        self.val_loss = float("inf")

    def aggregate(self, client_models, client_losses=None):
        if not client_models:
            return self.model

        # Convert to state_dict if necessary
        client_states = [
            m.state_dict() if hasattr(m, "state_dict") else m 
            for m in client_models
        ]

        global_dict = copy.deepcopy(client_states[0])
        num_clients = len(client_states)

        if self.update_type == "mse_avg":
            # If losses are provided and match count, compute inverse-loss weights
            if client_losses is not None and len(client_losses) == num_clients:
                epsilon = 1e-8
                inv_losses = [1.0 / (loss + epsilon) for loss in client_losses]
                total_inv_loss = sum(inv_losses)
                weights = [w / total_inv_loss for w in inv_losses]
            else:
                # Fallback to equal weighting if losses are missing (e.g., Quarantine / Time Buffer)
                weights = [1.0 / num_clients] * num_clients

            # Weighted aggregation
            for key in global_dict.keys():
                global_dict[key] = sum(
                    client_states[i][key].float() * weights[i]
                    for i in range(num_clients)
                )

        elif self.update_type in ["avg", "fedprox"]:
            # Standard FedAvg / FedProx parameter averaging
            for key in global_dict.keys():
                global_dict[key] = sum(
                    client_states[i][key].float() for i in range(num_clients)
                ) / num_clients

        else:
            raise ValueError(f"Unsupported update_type: {self.update_type}")

        # Load weights into global model
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(global_dict)
        else:
            self.model = global_dict

        return self.model
