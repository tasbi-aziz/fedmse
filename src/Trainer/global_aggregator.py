import copy
import logging
import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
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
        max_server_update_norm: float = 1.0,
        max_server_step_norm: float = 0.1
    ):
        """
        FedOpt / FedAdam Global Aggregator.

        Pipeline:

            Client Updates
                    ↓
            Effective Weight
                    ↓
            Weighted Pseudo-Gradient
                    ↓
            Pseudo-Gradient Clipping
                    ↓
            Adam First Moment (m_t)
                    ↓
            Adam Second Moment (v_t)
                    ↓
            Server Step Calculation
                    ↓
            FINAL Server Step Clipping
                    ↓
            Global Model Update

        Effective client weight:

            W_i = dataset_size_i × weight_factor_i

        NaN / Inf client model states are rejected.

        Suspicious but finite updates can still receive
        a low weight_factor such as 0.05.

        IMPORTANT:
        Low weight does NOT make NaN/Inf safe.
        Non-finite updates are rejected completely.
        """

        self.model = model

        # ---------------------------------------------------------
        # FEDOPT ONLY
        # ---------------------------------------------------------

        self.update_type = update_type.lower()

        if self.update_type != "fedopt":
            raise ValueError(
                "This GlobalAggregator is configured for "
                "FedOpt only. Use update_type='fedopt'."
            )

        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        # Maximum norm for aggregated pseudo-gradient
        self.max_server_update_norm = (
            max_server_update_norm
        )

        # Maximum norm for final Adam server step
        self.max_server_step_norm = (
            max_server_step_norm
        )

        # ---------------------------------------------------------
        # SERVER OPTIMIZER STATES
        # ---------------------------------------------------------

        self.m_t = {}
        self.v_t = {}

        self._init_server_moments()

    # =============================================================
    # STATE DICT
    # =============================================================

    def _get_state_dict(self) -> dict:
        """
        Extract current global model state_dict.
        """

        if hasattr(self.model, "state_dict"):
            return self.model.state_dict()

        return self.model

    # =============================================================
    # FINITE STATE CHECK
    # =============================================================

    def _state_is_finite(self, state: dict) -> bool:
        """
        Returns False if any floating-point tensor contains
        NaN or Inf.
        """

        if not isinstance(state, dict):
            return False

        for key, value in state.items():

            if isinstance(value, torch.Tensor):

                if value.is_floating_point():

                    if not torch.isfinite(value).all():
                        logging.error(
                            "[GlobalAggregator] "
                            f"NaN/Inf detected in parameter: {key}"
                        )
                        return False

        return True

    # =============================================================
    # INITIALIZE SERVER MOMENTS
    # =============================================================

    def _init_server_moments(self):
        """
        Initialize FedOpt Adam first and second moments.
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
        Returns a deep copy of the current global model.
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
    ) -> Union[torch.Tensor, None]:
        """
        Clip aggregated pseudo-gradient using L2 norm.

        Non-finite gradients are rejected.
        """

        if grad is None:
            return None

        # ---------------------------------------------------------
        # NaN / Inf protection
        # ---------------------------------------------------------

        if not torch.isfinite(grad).all():

            logging.error(
                "[GlobalAggregator] "
                "NaN/Inf detected in pseudo-gradient. "
                "Gradient rejected."
            )

            return None

        grad_norm = torch.norm(
            grad,
            p=2
        )

        if not torch.isfinite(grad_norm):

            logging.error(
                "[GlobalAggregator] "
                "Non-finite gradient norm detected. "
                "Gradient rejected."
            )

            return None

        # ---------------------------------------------------------
        # Gradient clipping
        # ---------------------------------------------------------

        if grad_norm > self.max_server_update_norm:

            scale = (
                self.max_server_update_norm
                /
                (grad_norm + 1e-12)
            )

            grad = grad * scale

        # Final safety check
        if not torch.isfinite(grad).all():
            logging.error(
                "[GlobalAggregator] "
                "Gradient became non-finite after clipping."
            )
            return None

        return grad

    # =============================================================
    # FINAL SERVER STEP CLIPPING
    # =============================================================

    def _clip_server_step(
        self,
        step: torch.Tensor
    ) -> Union[torch.Tensor, None]:
        """
        Final clipping of the FedOpt/Adam server step.

        This is applied AFTER:

            m_t
            v_t
            sqrt(v_t)
            server learning rate

        and BEFORE updating global parameters.
        """

        if step is None:
            return None

        # ---------------------------------------------------------
        # NaN / Inf protection
        # ---------------------------------------------------------

        if not torch.isfinite(step).all():

            logging.error(
                "[GlobalAggregator] "
                "NaN/Inf detected in server step. "
                "Step rejected."
            )

            return None

        step_norm = torch.norm(
            step,
            p=2
        )

        if not torch.isfinite(step_norm):

            logging.error(
                "[GlobalAggregator] "
                "Non-finite server step norm. "
                "Step rejected."
            )

            return None

        # ---------------------------------------------------------
        # Final server-step clipping
        # ---------------------------------------------------------

        if step_norm > self.max_server_step_norm:

            scale = (
                self.max_server_step_norm
                /
                (step_norm + 1e-12)
            )

            step = step * scale

        # ---------------------------------------------------------
        # Final safety check
        # ---------------------------------------------------------

        if not torch.isfinite(step).all():

            logging.error(
                "[GlobalAggregator] "
                "Server step became non-finite after clipping."
            )

            return None

        return step

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
        FedOpt aggregation.

        Effective client weight:

            W_i =
                dataset_size_i
                ×
                weight_factor_i

        NaN / Inf client states are rejected.

        Finite suspicious updates may receive reduced
        weight through weight_factor.
        """

        # =========================================================
        # SELECT UPDATE LIST
        # =========================================================

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
                "No client updates available."
            )

            return self.model

        extracted_states = []
        combined_weights = []

        rejected_clients = 0

        # =========================================================
        # 1. EXTRACT STATES + EFFECTIVE WEIGHTS
        # =========================================================

        for idx, item in enumerate(target_updates):

            # -----------------------------------------------------
            # Extract state
            # -----------------------------------------------------

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

                logging.warning(
                    "[GlobalAggregator] "
                    f"Unsupported client update #{idx}. "
                    "Rejected."
                )

                rejected_clients += 1
                continue

            # -----------------------------------------------------
            # CRITICAL NaN / Inf STATE CHECK
            # -----------------------------------------------------

            if not self._state_is_finite(state):

                logging.error(
                    "[GlobalAggregator] "
                    f"Client update #{idx} contains "
                    "NaN/Inf. UPDATE REJECTED."
                )

                rejected_clients += 1
                continue

            # -----------------------------------------------------
            # Dataset size
            # -----------------------------------------------------

            try:

                n_i = float(n_i)

                if not math.isfinite(n_i):
                    raise ValueError

                n_i = max(
                    n_i,
                    0.0
                )

            except (
                TypeError,
                ValueError
            ):

                logging.warning(
                    "[GlobalAggregator] "
                    f"Invalid dataset_size for client #{idx}. "
                    "Using 1.0."
                )

                n_i = 1.0

            # -----------------------------------------------------
            # Security weight
            # -----------------------------------------------------

            try:

                w_i = float(w_i)

                if not math.isfinite(w_i):
                    raise ValueError

                w_i = max(
                    w_i,
                    0.0
                )

            except (
                TypeError,
                ValueError
            ):

                logging.warning(
                    "[GlobalAggregator] "
                    f"Invalid weight_factor for client #{idx}. "
                    "Using 1.0."
                )

                w_i = 1.0

            # -----------------------------------------------------
            # Effective weight
            # -----------------------------------------------------

            effective_weight = (
                n_i * w_i
            )

            if not math.isfinite(
                effective_weight
            ):

                logging.error(
                    "[GlobalAggregator] "
                    f"Non-finite effective weight for "
                    f"client #{idx}. Rejected."
                )

                rejected_clients += 1
                continue

            if effective_weight <= 0:

                logging.warning(
                    "[GlobalAggregator] "
                    f"Client #{idx} has zero effective weight. "
                    "Skipped."
                )

                rejected_clients += 1
                continue

            # -----------------------------------------------------
            # Accept
            # -----------------------------------------------------

            extracted_states.append(
                state
            )

            combined_weights.append(
                effective_weight
            )

        # =========================================================
        # NO VALID CLIENTS
        # =========================================================

        if not extracted_states:

            logging.error(
                "[GlobalAggregator] "
                "No valid client updates remain. "
                "Global model unchanged."
            )

            return self.model

        # =========================================================
        # TOTAL WEIGHT
        # =========================================================

        total_weight = sum(
            combined_weights
        )

        if (
            total_weight <= 0
            or
            not math.isfinite(total_weight)
        ):

            logging.error(
                "[GlobalAggregator] "
                "Invalid total effective weight. "
                "Skipping aggregation."
            )

            return self.model

        num_clients = len(
            extracted_states
        )

        # =========================================================
        # 2. CURRENT GLOBAL MODEL
        # =========================================================

        current_global_state = (
            self._get_state_dict()
        )

        # ---------------------------------------------------------
        # Global model safety check
        # ---------------------------------------------------------

        if not self._state_is_finite(
            current_global_state
        ):

            logging.critical(
                "[GlobalAggregator] "
                "Current global model contains NaN/Inf. "
                "Aggregation aborted."
            )

            return self.model

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

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

        updated_global_state = (
            copy.deepcopy(
                current_global_state
            )
        )

        # =========================================================
        # 3. WEIGHTED PSEUDO-GRADIENT
        # =========================================================

        pseudo_gradients = {}

        for key in current_global_state.keys():

            param = current_global_state[key]

            # -----------------------------------------------------
            # Floating point parameters
            # -----------------------------------------------------

            if (
                isinstance(param, torch.Tensor)
                and param.is_floating_point()
            ):

                delta_sum = torch.zeros_like(
                    param,
                    dtype=torch.float32,
                    device=device
                )

                valid_delta_count = 0

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

                    # -------------------------------------------------
                    # Shape safety
                    # -------------------------------------------------

                    if (
                        client_param.shape
                        !=
                        global_param.shape
                    ):

                        logging.warning(
                            "[GlobalAggregator] "
                            f"Shape mismatch for parameter {key}. "
                            f"Client update skipped."
                        )

                        continue

                    # -------------------------------------------------
                    # Delta
                    # -------------------------------------------------

                    delta = (
                        client_param
                        -
                        global_param
                    )

                    # -------------------------------------------------
                    # Delta safety
                    # -------------------------------------------------

                    if not torch.isfinite(
                        delta
                    ).all():

                        logging.error(
                            "[GlobalAggregator] "
                            f"NaN/Inf delta detected "
                            f"for parameter {key}. "
                            "Client contribution skipped."
                        )

                        continue

                    delta_sum += (
                        delta
                        *
                        combined_weights[i]
                    )

                    valid_delta_count += 1

                # -------------------------------------------------
                # No valid contribution
                # -------------------------------------------------

                if valid_delta_count == 0:

                    logging.warning(
                        "[GlobalAggregator] "
                        f"No valid client contribution "
                        f"for parameter {key}. "
                        "Parameter unchanged."
                    )

                    pseudo_gradients[key] = None
                    continue

                # -------------------------------------------------
                # Weighted pseudo-gradient
                # -------------------------------------------------

                grad = (
                    delta_sum
                    /
                    total_weight
                )

                # -------------------------------------------------
                # Gradient safety + clipping
                # -------------------------------------------------

                grad = (
                    self._clip_server_gradient(
                        grad
                    )
                )

                if grad is None:

                    logging.error(
                        "[GlobalAggregator] "
                        f"Invalid pseudo-gradient for "
                        f"{key}. Parameter unchanged."
                    )

                    pseudo_gradients[key] = None
                    continue

                pseudo_gradients[key] = grad

        # =========================================================
        # 4. FEDOPT / FEDADAM SERVER UPDATE
        # =========================================================

        for key in current_global_state.keys():

            param = current_global_state[key]

            # -----------------------------------------------------
            # Floating point parameters
            # -----------------------------------------------------

            if (
                isinstance(param, torch.Tensor)
                and param.is_floating_point()
            ):

                grad = pseudo_gradients.get(
                    key,
                    None
                )

                # -------------------------------------------------
                # No valid gradient
                # -------------------------------------------------

                if grad is None:

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                grad = grad.to(
                    "cpu",
                    dtype=torch.float32
                )

                # -------------------------------------------------
                # Initialize moments if necessary
                # -------------------------------------------------

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

                # =================================================
                # FIRST MOMENT
                # =================================================

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
                # Moment safety
                # -------------------------------------------------

                if not torch.isfinite(
                    self.m_t[key]
                ).all():

                    logging.error(
                        "[GlobalAggregator] "
                        f"NaN/Inf in first moment "
                        f"for {key}. "
                        "Parameter update skipped."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                # =================================================
                # SECOND MOMENT
                # =================================================

                self.v_t[key] = (
                    self.beta2
                    *
                    self.v_t[key]
                    +
                    (1.0 - self.beta2)
                    *
                    torch.square(
                        grad
                    )
                )

                # -------------------------------------------------
                # Moment safety
                # -------------------------------------------------

                if not torch.isfinite(
                    self.v_t[key]
                ).all():

                    logging.error(
                        "[GlobalAggregator] "
                        f"NaN/Inf in second moment "
                        f"for {key}. "
                        "Parameter update skipped."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                # =================================================
                # ADAM DENOMINATOR
                # =================================================

                denom = (
                    torch.sqrt(
                        self.v_t[key]
                    )
                    +
                    self.tau
                )

                if not torch.isfinite(
                    denom
                ).all():

                    logging.error(
                        "[GlobalAggregator] "
                        f"Invalid denominator for {key}. "
                        "Parameter update skipped."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                # =================================================
                # SERVER STEP
                # =================================================

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
                # CRITICAL STEP SAFETY
                # -------------------------------------------------

                if not torch.isfinite(
                    step_update
                ).all():

                    logging.error(
                        "[GlobalAggregator] "
                        f"NaN/Inf server step for {key}. "
                        "Parameter update skipped."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                # =================================================
                # FINAL SERVER STEP CLIPPING
                # =================================================

                step_update = (
                    self._clip_server_step(
                        step_update
                    )
                )

                if step_update is None:

                    logging.error(
                        "[GlobalAggregator] "
                        f"Final server-step clipping "
                        f"failed for {key}. "
                        "Parameter update skipped."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                # =================================================
                # GLOBAL PARAMETER UPDATE
                # =================================================

                new_param = (
                    param.to(device)
                    +
                    step_update
                )

                # -------------------------------------------------
                # FINAL PARAMETER SAFETY
                # -------------------------------------------------

                if not torch.isfinite(
                    new_param
                ).all():

                    logging.critical(
                        "[GlobalAggregator] "
                        f"NaN/Inf detected in new global "
                        f"parameter {key}. "
                        "Keeping previous parameter."
                    )

                    updated_global_state[key] = (
                        param.clone()
                    )

                    continue

                updated_global_state[key] = (
                    new_param
                )

            # -----------------------------------------------------
            # Non-floating buffers
            # -----------------------------------------------------

            else:

                if key in extracted_states[0]:

                    updated_global_state[key] = (
                        extracted_states[0][key]
                        .to(device)
                    )

                else:

                    updated_global_state[key] = (
                        param.clone()
                    )

        # =========================================================
        # 5. FINAL GLOBAL MODEL SAFETY CHECK
        # =========================================================

        if not self._state_is_finite(
            updated_global_state
        ):

            logging.critical(
                "[GlobalAggregator] "
                "Final global model contains NaN/Inf. "
                "GLOBAL UPDATE ABORTED."
            )

            return self.model

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

        # =========================================================
        # LOGGING
        # =========================================================

        logging.info(
            "[GlobalAggregator] "
            f"Aggregated {num_clients} valid client updates "
            f"via FEDOPT | "
            f"Rejected: {rejected_clients} | "
            f"Total Effective Weight: "
            f"{total_weight:.2f}"
        )

        return self.model
