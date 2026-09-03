import copy
import torch


class GlobalAggregator:
    def __init__(self, model, update_type="avg"):
        self.model = model
        self.update_type = update_type
        self.val_loss = float("inf")

    def aggregate(self, client_models, client_losses=None):
        """
        Aggregates parameters from multiple client models into the global model.
        
        :param client_models: List of state_dict objects or model instances from clients.
        :param client_losses: List of MSE validation/training losses corresponding to each client.
                               Required when update_type is 'mse_avg'.
        """
        if not client_models:
            return self.model

        # Ensure we are working with state_dict representations
        client_states = [
            m.state_dict() if hasattr(m, "state_dict") else m 
            for m in client_models
        ]

        global_dict = copy.deepcopy(client_states[0])

        if self.update_type == "mse_avg":
            if client_losses is None or len(client_losses) != len(client_models):
                raise ValueError("client_losses list is required and must match client_models length for 'mse_avg'.")

            # Convert losses to inverse weights (lower MSE loss = higher weight)
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-8
            inv_losses = [1.0 / (loss + epsilon) for loss in client_losses]
            total_inv_loss = sum(inv_losses)
            weights = [w / total_inv_loss for w in inv_losses]

            # Weighted aggregation based on MSE performance
            for key in global_dict.keys():
                global_dict[key] = sum(
                    client_states[i][key].float() * weights[i]
                    for i in range(len(client_states))
                )

        elif self.update_type == "avg":
            # Standard FedAvg (unweighted average across all clients)
            num_clients = len(client_states)
            for key in global_dict.keys():
                global_dict[key] = sum(
                    client_states[i][key].float() for i in range(num_clients)
                ) / num_clients

        else:
            raise ValueError(f"Unsupported update_type: {self.update_type}")

        # Update global model parameters
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(global_dict)
        else:
            self.model = global_dict

        return self.model
