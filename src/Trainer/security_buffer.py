import copy
import logging
import math
import torch
import torch.nn as nn
from typing import List, Dict, Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SecurityBuffer:
    def __init__(
        self,
        global_model: nn.Module = None,
        window_size: int = 5,
        latency_threshold: float = 10.0,
        alpha: float = 0.2,
        beta: float = 0.01,
        cos_sim_threshold: float = -0.1,
        magnitude_threshold: float = 1.0,
        loss_change_threshold: float = 0.50,
        mse_change_threshold: float = 0.50,
        mse_std_change_threshold: float = 0.50,
        train_time_change_threshold: float = 0.50,
        history_size: int = 5,
        min_history: int = 2
    ):
        """
        Security Buffer for asynchronous federated learning.

        Current behavioral decision signals:
        1. Update magnitude
        2. Workload-normalized training-time behavior
        3. Local loss behavior
        4. Historical validation-MSE behavior

        Separately:
        - arrival_time > 10 sec routes the update to secondary
          processing, but lateness alone is NOT treated as an attack.

        BEFORE all behavioral checks:
        - The incoming update is checked for NaN / Inf.
        - Non-finite updates are rejected immediately.
        - NaN / Inf is NOT counted as one of the four
          behavioral security conditions.

        Cosine similarity is NO LONGER used for security decisions.
        It is retained in the constructor only for backward compatibility.

        Routing:
        - Fast + clean       -> DIRECT
        - Fast + suspicious  -> SECONDARY CHECK
        - Slow               -> SECONDARY CHECK
        - Secondary failure  -> QUARANTINE
        """

        self.global_model = (
            copy.deepcopy(global_model)
            if global_model is not None
            else None
        )

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.window_size = window_size
        self.latency_threshold = latency_threshold

        self.alpha = alpha
        self.beta = beta

        # Kept only so old code/config does not break.
        # NOT USED in any decision.
        self.cos_sim_threshold = cos_sim_threshold

        self.magnitude_threshold = magnitude_threshold
        self.loss_change_threshold = loss_change_threshold
        self.mse_change_threshold = mse_change_threshold
        self.mse_std_change_threshold = mse_std_change_threshold

        self.train_time_change_threshold = (
            train_time_change_threshold
        )

        self.history_size = history_size
        self.min_history = min_history

        # ---------------------------------------------------------
        # Delayed updates
        # ---------------------------------------------------------

        self.buffer: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Per-client historical behavior
        # ---------------------------------------------------------

        self.client_history: Dict[
            str,
            Dict[str, List[float]]
        ] = {}

    # =============================================================
    # GLOBAL MODEL
    # =============================================================

    def set_global_model(
        self,
        global_model: nn.Module
    ):
        """
        Update internal copy of global model.
        """

        if global_model is not None:

            self.global_model = copy.deepcopy(
                global_model
            )

    # =============================================================
    # FINITE UPDATE SAFETY CHECK
    # =============================================================

    def _is_update_finite(
        self,
        update: Dict[str, Any]
    ) -> bool:
        """
        Checks whether the incoming client update contains
        NaN or Inf values.

        This is a SAFETY check, NOT one of the four
        behavioral security conditions.

        Floating-point tensors are checked recursively.

        Returns:
            True  -> update is finite
            False -> update contains NaN/Inf
        """

        if not isinstance(update, dict):
            return False

        def check_value(value):

            # -----------------------------------------------------
            # Tensor
            # -----------------------------------------------------

            if isinstance(value, torch.Tensor):

                if value.is_floating_point():

                    return bool(
                        torch.isfinite(
                            value
                        ).all().item()
                    )

                return True

            # -----------------------------------------------------
            # Dictionary
            # -----------------------------------------------------

            if isinstance(value, dict):

                for nested_value in value.values():

                    if not check_value(
                        nested_value
                    ):
                        return False

                return True

            # -----------------------------------------------------
            # List / Tuple
            # -----------------------------------------------------

            if isinstance(
                value,
                (list, tuple)
            ):

                for nested_value in value:

                    if not check_value(
                        nested_value
                    ):
                        return False

                return True

            # -----------------------------------------------------
            # Numeric scalar
            # -----------------------------------------------------

            if isinstance(
                value,
                (float, int)
            ):

                if isinstance(
                    value,
                    float
                ):

                    return math.isfinite(
                        value
                    )

                return True

            # -----------------------------------------------------
            # Other metadata
            # -----------------------------------------------------

            return True

        return check_value(update)

    # =============================================================
    # CLIENT HISTORY
    # =============================================================

    def _get_client_history(
        self,
        client_id
    ):

        client_id = str(client_id)

        if client_id not in self.client_history:

            self.client_history[client_id] = {
                "magnitude": [],
                "train_time_per_sample": [],
                "val_loss": [],
                "mse_mean": [],
                "mse_std": [],
                "mse_max": []
            }

        return self.client_history[
            client_id
        ]

    def _append_history(
        self,
        client_id,
        magnitude,
        time_per_sample,
        val_loss,
        mse_mean,
        mse_std,
        mse_max
    ):

        history = self._get_client_history(
            client_id
        )

        values = {
            "magnitude": magnitude,
            "train_time_per_sample": time_per_sample,
            "val_loss": val_loss,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mse_max": mse_max
        }

        for key, value in values.items():

            if value is None:
                continue

            try:
                value = float(value)
            except (
                TypeError,
                ValueError
            ):
                continue

            if not math.isfinite(
                value
            ):
                continue

            history[key].append(
                value
            )

            if len(
                history[key]
            ) > self.history_size:

                history[key].pop(0)

    # =============================================================
    # UPDATE MAGNITUDE
    # =============================================================

    def compute_update_magnitude(
        self,
        model_update: dict,
        global_model: nn.Module
    ) -> float:

        if (
            global_model is None
            or not model_update
        ):
            return 0.0

        global_dict = (
            global_model.state_dict()
        )

        squared_sum = 0.0

        for name, local_param in (
            model_update.items()
        ):

            if name not in global_dict:
                continue

            global_param = (
                global_dict[name]
            )

            if not torch.is_floating_point(
                local_param
            ):
                continue

            if not torch.is_floating_point(
                global_param
            ):
                continue

            local_cpu = (
                local_param.detach()
                .cpu()
                .float()
            )

            global_cpu = (
                global_param.detach()
                .cpu()
                .float()
            )

            if (
                local_cpu.shape
                !=
                global_cpu.shape
            ):
                continue

            difference = (
                local_cpu
                -
                global_cpu
            )

            squared_sum += torch.sum(
                difference * difference
            ).item()

        return math.sqrt(
            max(
                squared_sum,
                0.0
            )
        )

    # =============================================================
    # MSE STATISTICS
    # =============================================================

    def compute_mse_statistics(
        self,
        val_mse_list: list
    ):

        if not val_mse_list:

            return {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0
            }

        safe_values = []

        for value in val_mse_list:

            try:

                value = float(
                    value
                )

                if not math.isfinite(
                    value
                ):
                    continue

                value = max(
                    value,
                    0.0
                )

                value = math.log1p(
                    value
                )

                safe_values.append(
                    value
                )

            except (
                TypeError,
                ValueError
            ):
                continue

        if not safe_values:

            return {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0
            }

        mse_tensor = torch.tensor(
            safe_values,
            dtype=torch.float32
        )

        return {
            "mean": float(
                torch.mean(
                    mse_tensor
                ).item()
            ),

            "std": float(
                torch.std(
                    mse_tensor,
                    unbiased=False
                ).item()
            ),

            "max": float(
                torch.max(
                    mse_tensor
                ).item()
            )
        }

    # =============================================================
    # ADAPTIVE THRESHOLD
    # =============================================================

    def _adaptive_threshold(
        self,
        history_values: List[float],
        base_threshold: float,
        factor: float = 3.0
    ):

        if not history_values:
            return base_threshold

        if (
            len(history_values)
            <
            self.min_history
        ):
            return base_threshold

        tensor = torch.tensor(
            history_values,
            dtype=torch.float32
        )

        mean_value = float(
            torch.mean(
                tensor
            ).item()
        )

        std_value = float(
            torch.std(
                tensor,
                unbiased=False
            ).item()
        )

        adaptive_value = (
            mean_value
            +
            factor * std_value
        )

        return max(
            base_threshold,
            adaptive_value
        )

    # =============================================================
    # WORKLOAD-AWARE TRAINING TIME
    # =============================================================

    def compute_time_per_sample(
        self,
        train_time: float,
        dataset_size: int
    ) -> float:

        try:

            train_time = float(
                train_time
            )

            dataset_size = int(
                dataset_size
            )

            if not math.isfinite(
                train_time
            ):
                return 0.0

            if train_time < 0:
                return 0.0

            if dataset_size <= 0:
                return 0.0

            return (
                train_time
                /
                dataset_size
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

    # =============================================================
    # BEHAVIOR CHECK
    # =============================================================

    def evaluate_update_behavior(
        self,
        update: Dict[str, Any],
        global_model: nn.Module = None
    ) -> Dict[str, Any]:

        if global_model is None:

            global_model = (
                self.global_model
            )

        client_id = update.get(
            "client_id",
            "unknown"
        )

        weights = update.get(
            "weights",
            {}
        )

        # ---------------------------------------------------------
        # arrival_time
        # ---------------------------------------------------------

        latency = update.get(
            "arrival_time",
            0.0
        )

        try:

            latency = float(
                latency
            )

        except (
            TypeError,
            ValueError
        ):

            latency = 0.0

        # ---------------------------------------------------------
        # train_time
        # ---------------------------------------------------------

        train_time = update.get(
            "train_time",
            0.0
        )

        try:

            train_time = float(
                train_time
            )

        except (
            TypeError,
            ValueError
        ):

            train_time = 0.0

        # ---------------------------------------------------------
        # dataset size
        # ---------------------------------------------------------

        dataset_size = update.get(
            "dataset_size",
            update.get(
                "data_size",
                0
            )
        )

        try:

            dataset_size = int(
                dataset_size
            )

        except (
            TypeError,
            ValueError
        ):

            dataset_size = 0

        # ---------------------------------------------------------
        # Local validation loss
        # ---------------------------------------------------------

        val_loss = update.get(
            "val_loss",
            0.0
        )

        try:

            val_loss = float(
                val_loss
            )

        except (
            TypeError,
            ValueError
        ):

            val_loss = 0.0

        # ---------------------------------------------------------
        # 5-fold MSE list
        # ---------------------------------------------------------

        val_mse_list = update.get(
            "val_mse_list",
            []
        )

        # =========================================================
        # 1. UPDATE MAGNITUDE
        # =========================================================

        magnitude = (
            self.compute_update_magnitude(
                weights,
                global_model
            )
        )

        # =========================================================
        # 2. WORKLOAD-NORMALIZED TRAINING TIME
        # =========================================================

        time_per_sample = (
            self.compute_time_per_sample(
                train_time,
                dataset_size
            )
        )

        # =========================================================
        # 3. MSE STATISTICS
        # =========================================================

        mse_stats = (
            self.compute_mse_statistics(
                val_mse_list
            )
        )

        # =========================================================
        # CLIENT HISTORY
        # =========================================================

        history = (
            self._get_client_history(
                client_id
            )
        )

        # =========================================================
        # CONDITION 1:
        # UPDATE MAGNITUDE
        # =========================================================

        magnitude_threshold = (
            self._adaptive_threshold(
                history["magnitude"],
                self.magnitude_threshold
            )
        )

        magnitude_fail = (
            magnitude
            >
            magnitude_threshold
            if history["magnitude"]
            else False
        )

        # =========================================================
        # CONDITION 2:
        # WORKLOAD-AWARE TIMING BEHAVIOR
        # =========================================================

        previous_time_per_sample = (
            history[
                "train_time_per_sample"
            ][-1]
            if history[
                "train_time_per_sample"
            ]
            else None
        )

        timing_change = 0.0

        if (
            previous_time_per_sample
            is not None
            and
            previous_time_per_sample > 0
            and
            time_per_sample >= 0
        ):

            timing_change = (
                abs(
                    time_per_sample
                    -
                    previous_time_per_sample
                )
                /
                (
                    abs(
                        previous_time_per_sample
                    )
                    +
                    1e-12
                )
            )

        timing_fail = (
            timing_change
            >
            self.train_time_change_threshold
            if previous_time_per_sample
            is not None
            else False
        )

        # ---------------------------------------------------------
        # 10-second arrival routing
        # ---------------------------------------------------------

        latency_fail = (
            latency
            >
            self.latency_threshold
        )

        # =========================================================
        # CONDITION 3:
        # LOCAL LOSS BEHAVIOR
        # =========================================================

        loss_threshold = (
            self._adaptive_threshold(
                history["val_loss"],
                self.loss_change_threshold
            )
        )

        loss_fail = False

        if (
            history["val_loss"]
            and
            val_loss > 0
        ):

            previous_loss = (
                history["val_loss"][-1]
            )

            if previous_loss > 0:

                relative_loss_change = (
                    abs(
                        val_loss
                        -
                        previous_loss
                    )
                    /
                    (
                        previous_loss
                        +
                        1e-8
                    )
                )

                loss_fail = (
                    relative_loss_change
                    >
                    loss_threshold
                )

        # =========================================================
        # CONDITION 4:
        # MSE HISTORY CHANGE
        # =========================================================

        mse_history_fail = False

        mse_mean_change = 0.0

        mse_std_change = 0.0

        if history["mse_mean"]:

            previous_mean = (
                history[
                    "mse_mean"
                ][-1]
            )

            previous_std = (
                history[
                    "mse_std"
                ][-1]
            )

            current_mean = (
                mse_stats["mean"]
            )

            current_std = (
                mse_stats["std"]
            )

            if previous_mean > 0:

                mse_mean_change = (
                    abs(
                        current_mean
                        -
                        previous_mean
                    )
                    /
                    (
                        abs(
                            previous_mean
                        )
                        +
                        1e-8
                    )
                )

            if previous_std > 0:

                mse_std_change = (
                    abs(
                        current_std
                        -
                        previous_std
                    )
                    /
                    (
                        abs(
                            previous_std
                        )
                        +
                        1e-8
                    )
                )

            mse_history_fail = (
                mse_mean_change
                >
                self.mse_change_threshold
                or
                mse_std_change
                >
                self.mse_std_change_threshold
            )

        # =========================================================
        # RISK / TRUST
        # =========================================================

        failed_conditions = sum([
            bool(magnitude_fail),
            bool(timing_fail),
            bool(loss_fail),
            bool(mse_history_fail)
        ])

        # =========================================================
        # NEW: INDIVIDUAL CONDITION DIAGNOSTIC LOG
        # =========================================================

        logging.info(
            f"[Security Buffer] Client {client_id} | "
            f"MagnitudeFail={magnitude_fail} | "
            f"TimingFail={timing_fail} | "
            f"LossFail={loss_fail} | "
            f"MSEHistoryFail={mse_history_fail}"
        )

        total_conditions = 4

        risk_score = (
            failed_conditions
            /
            total_conditions
        )

        trust_score = (
            1.0
            -
            risk_score
        )

        # =========================================================
        # DECISION
        # =========================================================

        if failed_conditions == 0:

            decision = "DIRECT"

            validation_required = False

        elif failed_conditions <= 2:

            decision = "SECONDARY_CHECK"

            validation_required = True

        else:

            decision = "QUARANTINE"

            validation_required = False

        # =========================================================
        # RETURN ALL DIAGNOSTIC VALUES
        # =========================================================

        return {

            "client_id":
                client_id,

            "magnitude":
                magnitude,

            "magnitude_threshold":
                magnitude_threshold,

            "latency":
                latency,

            "train_time":
                train_time,

            "dataset_size":
                dataset_size,

            "time_per_sample":
                time_per_sample,

            "previous_time_per_sample":
                previous_time_per_sample,

            "timing_change":
                timing_change,

            "time_change_threshold":
                self.train_time_change_threshold,

            "val_loss":
                val_loss,

            "loss_threshold":
                loss_threshold,

            "mse_mean":
                mse_stats["mean"],

            "mse_std":
                mse_stats["std"],

            "mse_max":
                mse_stats["max"],

            "mse_mean_change":
                mse_mean_change,

            "mse_std_change":
                mse_std_change,

            "Magnitude_Fail":
                magnitude_fail,

            "Latency_Fail":
                latency_fail,

            "Timing_Fail":
                timing_fail,

            "Loss_Fail":
                loss_fail,

            "MSE_History_Fail":
                mse_history_fail,

            "failed_conditions":
                failed_conditions,

            "risk_score":
                risk_score,

            "trust_score":
                trust_score,

            "score":
                trust_score,

            "decision":
                decision,

            "validation_required":
                validation_required
        }

    # =============================================================
    # COLLECT CURRENT ROUND UPDATES
    # =============================================================

    def collect_current_round_updates(
        self,
        incoming_updates: List[Dict[str, Any]],
        global_model: nn.Module = None,
        val_loader=None,
        criterion=None,
        global_mse: float = None,
        device: str = "cpu",
        **kwargs
    ) -> List[Dict[str, Any]]:

        if global_model is not None:

            self.set_global_model(
                global_model
            )

        ready_updates = []

        next_buffer = []

        # =========================================================
        # STEP 1:
        # RECHECK EXISTING BUFFERED UPDATES
        # =========================================================

        for buffered_item in self.buffer:

            buffered_item["age"] += 1

            client_id = (
                buffered_item.get(
                    "client_id",
                    "unknown"
                )
            )

            # -----------------------------------------------------
            # FINITE SAFETY CHECK
            # -----------------------------------------------------

            if not self._is_update_finite(
                buffered_item
            ):

                logging.warning(
                    f"[Security Buffer] "
                    f"REJECTED non-finite buffered update | "
                    f"Client: {client_id} | "
                    f"NaN/Inf detected"
                )

                continue

            # -----------------------------------------------------
            # Expiration
            # -----------------------------------------------------

            if (
                buffered_item["age"]
                >
                self.window_size
            ):

                logging.warning(
                    f"[Security Buffer] DROPPED expired update | "
                    f"Client: {client_id} | "
                    f"Reached Max Age ({self.window_size})"
                )

                continue

            # -----------------------------------------------------
            # Secondary checking
            # -----------------------------------------------------

            behavior = (
                self.evaluate_update_behavior(
                    buffered_item,
                    self.global_model
                )
            )

            buffered_item.update(
                behavior
            )

            # -----------------------------------------------------
            # Secondary check passed
            # -----------------------------------------------------

            if (
                behavior["decision"]
                ==
                "DIRECT"
            ):

                buffered_item["route"] = (
                    "RELEASED"
                )

                buffered_item[
                    "validation_required"
                ] = False

                ready_updates.append(
                    buffered_item
                )

                self._remember_update(
                    buffered_item
                )

                logging.info(
                    f"[Security Buffer] "
                    f"RELEASED buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Arrival Latency: "
                    f"{behavior['latency']:.2f}s | "
                    f"Train Time: "
                    f"{behavior['train_time']:.2f}s | "
                    f"Time/Sample: "
                    f"{behavior['time_per_sample']:.6f} | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # Still suspicious
            # -----------------------------------------------------

            elif (
                behavior["decision"]
                ==
                "SECONDARY_CHECK"
            ):

                next_buffer.append(
                    buffered_item
                )

                logging.info(
                    f"[Security Buffer] "
                    f"HOLDING buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # Strongly suspicious
            # -----------------------------------------------------

            else:

                buffered_item[
                    "route"
                ] = "QUARANTINE"

                logging.warning(
                    f"[Security Buffer] "
                    f"QUARANTINE buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Failed: "
                    f"{behavior['failed_conditions']}/4"
                )

        # =========================================================
        # STEP 2:
        # PROCESS NEW INCOMING UPDATES
        # =========================================================

        for update in incoming_updates:

            client_id = (
                update.get(
                    "client_id",
                    "unknown"
                )
            )

            # -----------------------------------------------------
            # FINITE SAFETY CHECK
            #
            # This happens BEFORE the four behavioral checks.
            # NaN/Inf is NOT counted in failed_conditions.
            # -----------------------------------------------------

            if not self._is_update_finite(
                update
            ):

                update[
                    "route"
                ] = "REJECT"

                update[
                    "validation_required"
                ] = False

                logging.warning(
                    f"[Security Buffer] "
                    f"REJECT Client {client_id} | "
                    f"NaN/Inf detected in incoming update"
                )

                continue

            # -----------------------------------------------------
            # Existing behavioral security logic
            # -----------------------------------------------------

            behavior = (
                self.evaluate_update_behavior(
                    update,
                    self.global_model
                )
            )

            update.update(
                behavior
            )

            latency = (
                behavior["latency"]
            )

            # -----------------------------------------------------
            # FAST + CLEAN
            # -----------------------------------------------------

            if (
                latency
                <=
                self.latency_threshold
                and
                behavior["decision"]
                ==
                "DIRECT"
            ):

                update["route"] = (
                    "DIRECT"
                )

                update[
                    "validation_required"
                ] = False

                ready_updates.append(
                    update
                )

                self._remember_update(
                    update
                )

                logging.info(
                    f"[Security Buffer] "
                    f"DIRECT ACCEPT Client {client_id} | "
                    f"Arrival Latency: "
                    f"{latency:.2f}s | "
                    f"Train Time: "
                    f"{behavior['train_time']:.2f}s | "
                    f"Time/Sample: "
                    f"{behavior['time_per_sample']:.6f} | "
                    f"Magnitude: "
                    f"{behavior['magnitude']:.4f} | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # FAST BUT SUSPICIOUS
            # -----------------------------------------------------

            elif (
                latency
                <=
                self.latency_threshold
                and
                behavior["decision"]
                ==
                "SECONDARY_CHECK"
            ):

                update_copy = (
                    copy.deepcopy(
                        update
                    )
                )

                update_copy["age"] = 0

                update_copy[
                    "route"
                ] = "SECONDARY_CHECK"

                update_copy[
                    "validation_required"
                ] = True

                next_buffer.append(
                    update_copy
                )

                logging.info(
                    f"[Security Buffer] "
                    f"SECONDARY CHECK Client {client_id} | "
                    f"Arrival Latency: "
                    f"{latency:.2f}s | "
                    f"Train Time: "
                    f"{behavior['train_time']:.2f}s | "
                    f"Time/Sample: "
                    f"{behavior['time_per_sample']:.6f} | "
                    f"Failed: "
                    f"{behavior['failed_conditions']}/4 | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # SLOW / LATE UPDATE
            # -----------------------------------------------------

            elif (
                latency
                >
                self.latency_threshold
            ):

                update_copy = (
                    copy.deepcopy(
                        update
                    )
                )

                update_copy["age"] = 0

                update_copy[
                    "route"
                ] = "SECONDARY_CHECK"

                update_copy[
                    "validation_required"
                ] = True

                next_buffer.append(
                    update_copy
                )

                logging.info(
                    f"[Security Buffer] "
                    f"BUFFERED SLOW Client {client_id} | "
                    f"Arrival Latency: "
                    f"{latency:.2f}s > "
                    f"{self.latency_threshold:.2f}s | "
                    f"Train Time: "
                    f"{behavior['train_time']:.2f}s | "
                    f"Time/Sample: "
                    f"{behavior['time_per_sample']:.6f} | "
                    f"Timing Fail: "
                    f"{behavior['Timing_Fail']} | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # STRONGLY SUSPICIOUS
            # -----------------------------------------------------

            else:

                update[
                    "route"
                ] = "QUARANTINE"

                update[
                    "validation_required"
                ] = False

                logging.warning(
                    f"[Security Buffer] "
                    f"QUARANTINE Client {client_id} | "
                    f"Failed: "
                    f"{behavior['failed_conditions']}/4 | "
                    f"Trust: "
                    f"{behavior['trust_score']:.3f}"
                )

        # =========================================================
        # UPDATE BUFFER
        # =========================================================

        self.buffer = next_buffer

        logging.info(
            f"[Security Buffer] Summary -> "
            f"Direct/Released Updates: "
            f"{len(ready_updates)} | "
            f"Secondary/Buffered: "
            f"{len(self.buffer)}"
        )

        return ready_updates

    # =============================================================
    # STORE HISTORY
    # =============================================================

    def _remember_update(
        self,
        update
    ):

        client_id = (
            update.get(
                "client_id",
                "unknown"
            )
        )

        self._append_history(
            client_id=client_id,

            magnitude=update.get(
                "magnitude"
            ),

            time_per_sample=update.get(
                "time_per_sample"
            ),

            val_loss=update.get(
                "val_loss"
            ),

            mse_mean=update.get(
                "mse_mean"
            ),

            mse_std=update.get(
                "mse_std"
            ),

            mse_max=update.get(
                "mse_max"
            )
        )

    # =============================================================
    # DIAGNOSTIC STATUS
    # =============================================================

    def get_buffer_status(
        self
    ) -> dict:

        return {

            "buffered_count":
                len(
                    self.buffer
                ),

            "buffered_clients": [
                item.get(
                    "client_id"
                )
                for item in self.buffer
            ],

            "window_size":
                self.window_size,

            "latency_threshold":
                self.latency_threshold,

            "history_clients":
                len(
                    self.client_history
                )
        }
