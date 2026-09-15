import copy
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GlobalAggregator:

    def __init__(
        self,
        model: Union[nn.Module, dict],
        update_type: str = "fedopt",
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3,
        max_server_update_norm: float = 1.0
    ):
        """
        Global Aggregator for FedMSE using:

        - FedOpt / FedAdam
        - Weighted FedAvg

        Effective client weight:

            W_i = dataset_size * weight_factor

        Server-side pseudo-gradient clipping is applied to
        prevent extreme aggregated updates from destabilizing
        the global model.

        Parameters
        ----------
        model:
            Global PyTorch model or state_dict.

        update_type:
            'fedopt', 'fedadam', or 'fedavg'.

        server_lr:
            Server-side learning rate.

        beta1:
            First-moment decay.

        beta2:
            Second-moment decay.

        tau:
            Numerical stability term.

        max_server_update_norm:
            Maximum norm allowed for each server-side
            pseudo-gradient tensor.
        """

        self.model = model

        self.update_type = update_type.lower()

        self.server_lr = server_lr

        self.beta1 = beta1

        self.beta2 = beta2

        self.tau = tau

        self.max_server_update_norm = (
            max_server_update_norm
        )

        # ---------------------------------------------------------
        # Server Optimizer States
        # ---------------------------------------------------------

        self.m_t = {}

        self.v_t = {}

        self._init_server_moments()

    # =============================================================
    # STATE DICT
    # =============================================================

    def _get_state_dict(self) -> dict:
        """
        Extract model state_dict.
        """

        if hasattr(
            self.model,
            "state_dict"
        ):
            return self.model.state_dict()

        return self.model

    # =============================================================
    # INITIALIZE SERVER MOMENTS
    # =============================================================

    def _init_server_moments(self):
        """
        Initializes first and second moment buffers.
        """

        state_dict = self._get_state_dict()

        for key, param in state_dict.items():

            if (
                isinstance(param, torch.Tensor)
                and param.is_floating_point()
            ):

                self.m_t[key] = torch.zeros_like(
                    param,
                    dtype=torch.float32,
                    device="cpu"
                )

                self.v_t[key] = torch.zeros_like(
                    param,
                    dtype=torch.float32,
                    device="cpu"
                )

    # =============================================================
    # GET GLOBAL PARAMETERS
    # =============================================================

    def get_global_parameters(self) -> dict:
        """
        Returns current global model parameters.
        """

        return copy.deepcopy(
            self._get_state_dict()
        )

    # =============================================================
    # SERVER PSEUDO-GRADIENT CLIPPING
    # =============================================================

    def _clip_server_gradient(
        self,
        grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Clips the aggregated server pseudo-gradient
        using an L2 norm.

        This protects FedOpt from unusually large
        aggregated client updates.
        """

        if grad is None:
            return grad

        grad_norm = torch.norm(
            grad,
            p=2
        )

        if (
            torch.isfinite(grad_norm)
            and
            grad_norm > self.max_server_update_norm
        ):

            scale = (
                self.max_server_update_norm
                /
                (grad_norm + 1e-12)
            )

            grad = grad * scale

        return grad

    # =============================================================
    # AGGREGATE
    # =============================================================

    def aggregate(
        self,
        client_updates: List[Dict[str, Any]] = None,
        client_models: List[Any] = None,
        updates: List[Any] = None,
        **kwargs
    ) -> Union[nn.Module, dict]:

        """
        Aggregates client updates.

        Effective weight:

            W_i =
                dataset_size_i
                ×
                weight_factor_i

        where weight_factor_i is produced by
        the security / validation stage.
        """

        target_updates = (
            client_updates
            if client_updates is not None
            else (
                client_models
                if client_models is not None
                else updates
            )
        )

        if not target_updates:

            logging.warning(
                "[GlobalAggregator] "
                "No client updates available for aggregation."
            )

            return self.model

        extracted_states = []

        combined_weights = []

        # =========================================================
        # 1. EXTRACT STATES + EFFECTIVE WEIGHTS
        # =========================================================

        for item in target_updates:

            if (
                isinstance(item, dict)
                and "weights" in item
            ):

                state = item["weights"]

                n_i = item.get(
                    "dataset_size",
                    1
                )

                w_i = item.get(
                    "weight_factor",
                    1.0
                )

            elif hasattr(
                item,
                "state_dict"
            ):

                state = item.state_dict()

                n_i = getattr(
                    item,
                    "dataset_size",
                    1
                )

                w_i = getattr(
                    item,
                    "weight_factor",
                    1.0
                )

            elif isinstance(
                item,
                dict
            ):

                state = item

                n_i = 1

                w_i = 1.0

            else:

                raise ValueError(
                    f"Unsupported format in client_updates item: "
                    f"{type(item)}"
                )

            # -----------------------------------------------------
            # Safety check
            # -----------------------------------------------------

            try:
                n_i = max(
                    float(n_i),
                    0.0
                )
            except (
                TypeError,
                ValueError
            ):
                n_i = 1.0

            try:
                w_i = max(
                    float(w_i),
                    0.0
                )
            except (
                TypeError,
                ValueError
            ):
                w_i = 1.0

            effective_weight = (
                n_i * w_i
            )

            extracted_states.append(
                state
            )

            combined_weights.append(
                effective_weight
            )

        num_clients = len(
            extracted_states
        )

        total_weight = sum(
            combined_weights
        )

        if total_weight <= 0:

            logging.error(
                "[GlobalAggregator] "
                "Total effective weight is 0. "
                "Skipping aggregation."
            )

            return self.model

        # =========================================================
        # 2. CURRENT GLOBAL MODEL
        # =========================================================

        current_global_state = (
            self._get_state_dict()
        )

        if hasattr(
            self.model,
            "parameters"
        ):

            try:
                device = next(
                    self.model.parameters()
                ).device

            except StopIteration:

                device = torch.device(
                    "cpu"
                )

        else:

            device = torch.device(
                "cpu"
            )

        updated_global_state = copy.deepcopy(
            current_global_state
        )

        # =========================================================
        # 3. WEIGHTED PSEUDO-GRADIENT
        # =========================================================

        pseudo_gradients = {}

        for key in current_global_state.keys():

            param = current_global_state[key]

            if (
                isinstance(param, torch.Tensor)
                and param.is_floating_point()
            ):

                delta_sum = torch.zeros_like(
                    param,
                    dtype=torch.float32,
                    device=device
                )

                for i in range(
                    num_clients
                ):

                    if key not in extracted_states[i]:
                        continue

                    client_param = (
                        extracted_states[i][key]
                        .to(device)
                        .float()
                    )

                    global_param = (
                        param
                        .to(device)
                        .float()
                    )

                    if (
                        client_param.shape
                        !=
                        global_param.shape
                    ):
                        continue

                    delta = (
                        client_param
                        -
                        global_param
                    )

                    delta_sum += (
                        delta
                        *
                        combined_weights[i]
                    )

                grad = (
                    delta_sum
                    /
                    total_weight
                )

                # -------------------------------------------------
                # SERVER-SIDE GRADIENT CLIPPING
                # -------------------------------------------------

                grad = self._clip_server_gradient(
                    grad
                )

                pseudo_gradients[key] = grad

        # =========================================================
        # 4. FEDOPT / FEDADAM
        # =========================================================

        if self.update_type in [
            "fedopt",
            "fedadam"
        ]:

            for key in current_global_state.keys():

                param = current_global_state[key]

                if (
                    isinstance(param, torch.Tensor)
                    and param.is_floating_point()
                ):

                    grad = pseudo_gradients[key].to(
                        "cpu"
                    )

                    if key not in self.m_t:

                        self.m_t[key] = (
                            torch.zeros_like(
                                param,
                                dtype=torch.float32,
                                device="cpu"
                            )
                        )

                        self.v_t[key] = (
                            torch.zeros_like(
                                param,
                                dtype=torch.float32,
                                device="cpu"
                            )
                        )

                    # -------------------------------------------------
                    # First moment
                    # -------------------------------------------------

                    self.m_t[key] = (
                        self.beta1
                        *
                        self.m_t[key]
                        +
                        (1.0 - self.beta1)
                        *
                        grad
                    )

                    # -------------------------------------------------
                    # Second moment
                    # -------------------------------------------------

                    self.v_t[key] = (
                        self.beta2
                        *
                        self.v_t[key]
                        +
                        (1.0 - self.beta2)
                        *
                        torch.square(grad)
                    )

                    # -------------------------------------------------
                    # Numerical protection
                    # -------------------------------------------------

                    denom = (
                        torch.sqrt(
                            self.v_t[key]
                        )
                        +
                        self.tau
                    )

                    step_update = (
                        self.server_lr
                        *
                        (
                            self.m_t[key]
                            /
                            denom
                        )
                    ).to(device)

                    # -------------------------------------------------
                    # Global update
                    # -------------------------------------------------

                    updated_global_state[key] = (
                        param.to(device)
                        +
                        step_update
                    )

                else:

                    updated_global_state[key] = (
                        extracted_states[0][key]
                        .to(device)
                    )

        # =========================================================
        # 5. WEIGHTED FEDAVG
        # =========================================================

        else:

            for key in current_global_state.keys():

                param = current_global_state[key]

                if (
                    isinstance(param, torch.Tensor)
                    and param.is_floating_point()
                ):

                    updated_global_state[key] = (
                        param.to(device)
                        +
                        pseudo_gradients[key]
                    )

                else:

                    updated_global_state[key] = (
                        extracted_states[0][key]
                        .to(device)
                    )

        # =========================================================
        # 6. LOAD GLOBAL MODEL
        # =========================================================

        if hasattr(
            self.model,
            "load_state_dict"
        ):

            self.model.load_state_dict(
                updated_global_state
            )

        else:

            self.model = (
                updated_global_state
            )

        logging.info(
            f"[GlobalAggregator] "
            f"Aggregated {num_clients} client updates "
            f"via {self.update_type.upper()} | "
            f"Total Effective Weight: "
            f"{total_weight:.2f}"
        )

        return self.model
