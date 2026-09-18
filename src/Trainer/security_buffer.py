# security_buffer.py

import copy
import math
import logging
from typing import List, Dict, Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class SecurityBuffer:
    """
    Security-aware asynchronous / latency-aware update router.

    Routing policy
    --------------

    Safety check:
        NaN / Inf
            -> DROP

    Four behavioral security conditions:
        1. Update magnitude
        2. Workload-normalized training-time behavior
        3. Local validation-loss behavior
        4. Historical validation-MSE behavior

    Latency is NOT a security failure.

    Routing:

        3-4 security failures
            -> QUARANTINE
            -> buffer for next round
            -> main.py verifies later
            -> accepted => factor 0.3
            -> rejected => DROP

        1-2 security failures
            -> SECONDARY BUFFER
            -> buffer for next round
            -> main.py verifies later
            -> accepted => factor 0.7
            -> rejected => DROP

        0 security failures:

            FAST
                arrival_latency <= 1.5 sec
                -> DIRECT
                -> current round
                -> factor 1.0

            SLOW
                arrival_latency > 1.5 sec
                -> SECONDARY BUFFER
                -> next round
                -> factor 0.7


    IMPORTANT:

    This class only decides the routing of CURRENT-ROUND updates.

    Buffered updates are NOT re-validated or released here.

    main.py is responsible for:
        - validating buffered updates
        - accepting/rejecting them
        - carrying accepted updates to next-round aggregation
        - applying factor 0.7 / 0.3
    """

    def __init__(
        self,
        global_model: nn.Module = None,
        window_size: int = 5,
        latency_threshold: float = 1.5,
        alpha: float = 0.2,
        beta: float = 0.01,
        cos_sim_threshold: float = -0.1,
        magnitude_threshold: float = 1.0,
        loss_change_threshold: float = 0.50,
        mse_change_threshold: float = 0.50,
        mse_std_change_threshold: float = 0.50,
        train_time_change_threshold: float = 0.50,
        history_size: int = 5,
        min_history: int = 2,
    ):

        self.global_model = global_model

        # Maximum buffer age.
        self.window_size = window_size

        # FAST / SLOW threshold.
        self.latency_threshold = latency_threshold

        self.alpha = alpha
        self.beta = beta

        # Kept for backward compatibility.
        self.cos_sim_threshold = cos_sim_threshold

        # Security thresholds.
        self.magnitude_threshold = magnitude_threshold
        self.loss_change_threshold = loss_change_threshold
        self.mse_change_threshold = mse_change_threshold
        self.mse_std_change_threshold = mse_std_change_threshold
        self.train_time_change_threshold = train_time_change_threshold

        # Historical behavior settings.
        self.history_size = history_size
        self.min_history = min_history

        # Delayed updates.
        #
        # IMPORTANT:
        # These updates are NOT current-round aggregation updates.
        #
        # main.py will process this buffer in the next-round logic.
        self.buffer: List[Dict[str, Any]] = []

        # Per-client behavioral history.
        self.client_history: Dict[
            str,
            Dict[str, List[float]]
        ] = {}

    # ============================================================
    # GLOBAL MODEL
    # ============================================================

    def set_global_model(
        self,
        global_model: nn.Module
    ):
        """
        Update the current global model reference.
        """
        self.global_model = global_model

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    def _is_update_finite(
        self,
        obj: Any
    ) -> bool:
        """
        Recursively check whether an object contains NaN / Inf.
        """

        if torch.is_tensor(obj):
            return bool(
                torch.isfinite(obj).all().item()
            )

        if isinstance(obj, dict):
            return all(
                self._is_update_finite(value)
                for value in obj.values()
            )

        if isinstance(obj, (list, tuple)):
            return all(
                self._is_update_finite(value)
                for value in obj
            )

        if isinstance(obj, (float, int)):
            return math.isfinite(
                float(obj)
            )

        return True

    # ============================================================
    # CLIENT HISTORY
    # ============================================================

    def _get_client_history(
        self,
        client_id: str
    ):

        if client_id not in self.client_history:

            self.client_history[client_id] = {
                "magnitude": [],
                "train_time_per_sample": [],
                "val_loss": [],
                "mse_mean": [],
                "mse_std": [],
                "mse_max": [],
            }

        return self.client_history[client_id]

    def _append_history(
        self,
        client_id: str,
        key: str,
        value: float,
    ):

        if value is None:
            return

        try:
            value = float(value)
        except Exception:
            return

        if not math.isfinite(value):
            return

        history = self._get_client_history(
            client_id
        )

        history[key].append(value)

        if len(history[key]) > self.history_size:

            history[key] = (
                history[key][-self.history_size:]
            )

    def _remember_update(
        self,
        update: Dict[str, Any]
    ):

        client_id = str(
            update.get(
                "client_id",
                "unknown"
            )
        )

        diagnostics = update.get(
            "security_diagnostics",
            {}
        )

        self._append_history(
            client_id,
            "magnitude",
            diagnostics.get("magnitude"),
        )

        self._append_history(
            client_id,
            "train_time_per_sample",
            diagnostics.get(
                "train_time_per_sample"
            ),
        )

        self._append_history(
            client_id,
            "val_loss",
            diagnostics.get("val_loss"),
        )

        self._append_history(
            client_id,
            "mse_mean",
            diagnostics.get("mse_mean"),
        )

        self._append_history(
            client_id,
            "mse_std",
            diagnostics.get("mse_std"),
        )

        self._append_history(
            client_id,
            "mse_max",
            diagnostics.get("mse_max"),
        )

    # ============================================================
    # UPDATE MAGNITUDE
    # ============================================================

    def compute_update_magnitude(
        self,
        update_weights: Dict[str, torch.Tensor],
    ) -> float:

        if self.global_model is None:
            return 0.0

        global_state = (
            self.global_model.state_dict()
        )

        total_sq = 0.0

        for name, local_tensor in update_weights.items():

            if name not in global_state:
                continue

            global_tensor = global_state[name]

            try:

                local_tensor = (
                    local_tensor.detach().float()
                )

                global_tensor = (
                    global_tensor.detach().float()
                )

                if (
                    local_tensor.shape
                    != global_tensor.shape
                ):
                    continue

                diff = (
                    local_tensor
                    - global_tensor
                )

                total_sq += float(
                    torch.sum(
                        diff * diff
                    ).item()
                )

            except Exception:
                continue

        return math.sqrt(
            max(total_sq, 0.0)
        )

    # ============================================================
    # MSE STATISTICS
    # ============================================================

    def compute_mse_statistics(
        self,
        mse_list,
    ):

        if mse_list is None:
            return 0.0, 0.0, 0.0

        finite_values = []

        for value in mse_list:

            try:

                value = float(value)

                if math.isfinite(value):

                    value = max(
                        value,
                        0.0
                    )

                    finite_values.append(
                        value
                    )

            except Exception:
                continue

        if not finite_values:
            return 0.0, 0.0, 0.0

        # log1p prevents extremely large
        # MSE values from dominating history.
        values = torch.tensor(
            finite_values,
            dtype=torch.float32,
        )

        values = torch.log1p(values)

        mean_value = float(
            torch.mean(values).item()
        )

        if len(values) > 1:

            std_value = float(
                torch.std(
                    values,
                    unbiased=False,
                ).item()
            )

        else:

            std_value = 0.0

        max_value = float(
            torch.max(values).item()
        )

        return (
            mean_value,
            std_value,
            max_value,
        )

    # ============================================================
    # ADAPTIVE THRESHOLD
    # ============================================================

    def _adaptive_threshold(
        self,
        history_values: List[float],
        base_threshold: float,
    ) -> float:

        if (
            len(history_values)
            < self.min_history
        ):
            return base_threshold

        try:

            values = torch.tensor(
                history_values,
                dtype=torch.float32,
            )

            mean = float(
                torch.mean(values).item()
            )

            std = float(
                torch.std(
                    values,
                    unbiased=False,
                ).item()
            )

            adaptive = (
                mean + (3.0 * std)
            )

            return min(
                float(base_threshold),
                float(adaptive),
            )

        except Exception:

            return base_threshold

    # ============================================================
    # WORKLOAD NORMALIZED TRAINING TIME
    # ============================================================

    def compute_time_per_sample(
        self,
        train_time: float,
        dataset_size: int,
    ) -> float:

        try:

            train_time = float(
                train_time
            )

            dataset_size = int(
                dataset_size
            )

            if dataset_size <= 0:
                return 0.0

            if not math.isfinite(
                train_time
            ):
                return 0.0

            return (
                train_time
                / float(dataset_size)
            )

        except Exception:

            return 0.0

    # ============================================================
    # RELATIVE CHANGE
    # ============================================================

    @staticmethod
    def _relative_change(
        current: float,
        previous: float,
    ) -> float:

        try:

            current = float(current)
            previous = float(previous)

            denominator = max(
                abs(previous),
                1e-12,
            )

            return abs(
                current - previous
            ) / denominator

        except Exception:

            return 0.0

    # ============================================================
    # BEHAVIOR EVALUATION
    # ============================================================

    def evaluate_update_behavior(
        self,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:

        client_id = str(
            update.get(
                "client_id",
                "unknown"
            )
        )

        weights = update.get(
            "weights",
            {}
        )

        arrival_latency = float(
            update.get(
                "arrival_time",
                update.get(
                    "latency",
                    0.0
                ),
            )
        )

        train_time = float(
            update.get(
                "train_time",
                0.0
            )
        )

        dataset_size = int(
            update.get(
                "dataset_size",
                1
            )
        )

        val_loss = float(
            update.get(
                "val_loss",
                0.0
            )
        )

        mse_list = update.get(
            "val_mse_list",
            []
        )

        # --------------------------------------------------------
        # Basic finite checks
        # --------------------------------------------------------

        if not math.isfinite(
            arrival_latency
        ):
            arrival_latency = float(
                "inf"
            )

        if not math.isfinite(
            train_time
        ):
            train_time = float(
                "inf"
            )

        if not math.isfinite(
            val_loss
        ):
            val_loss = float(
                "inf"
            )

        # --------------------------------------------------------
        # Current measurements
        # --------------------------------------------------------

        magnitude = (
            self.compute_update_magnitude(
                weights
            )
        )

        time_per_sample = (
            self.compute_time_per_sample(
                train_time,
                dataset_size,
            )
        )

        (
            mse_mean,
            mse_std,
            mse_max,
        ) = self.compute_mse_statistics(
            mse_list
        )

        history = (
            self._get_client_history(
                client_id
            )
        )

        # ========================================================
        # CONDITION 1
        # UPDATE MAGNITUDE
        # ========================================================

        magnitude_fail = False

        if (
            len(history["magnitude"])
            >= self.min_history
        ):

            magnitude_threshold = (
                self._adaptive_threshold(
                    history["magnitude"],
                    self.magnitude_threshold,
                )
            )

            magnitude_fail = (
                magnitude
                > magnitude_threshold
            )

        else:

            magnitude_threshold = (
                self.magnitude_threshold
            )

        # ========================================================
        # CONDITION 2
        # WORKLOAD NORMALIZED TRAINING TIME
        # ========================================================

        timing_fail = False

        if (
            len(
                history[
                    "train_time_per_sample"
                ]
            )
            >= self.min_history
        ):

            previous_time = (
                history[
                    "train_time_per_sample"
                ][-1]
            )

            timing_change = (
                self._relative_change(
                    time_per_sample,
                    previous_time,
                )
            )

            timing_fail = (
                timing_change
                > self.train_time_change_threshold
            )

        else:

            timing_change = 0.0

        # ========================================================
        # LATENCY CLASSIFICATION
        # ========================================================

        # IMPORTANT:
        # Latency is NOT a security failure.

        latency_fail = (
            arrival_latency
            > self.latency_threshold
        )

        if latency_fail:
            latency_class = "SLOW"
        else:
            latency_class = "FAST"

        # ========================================================
        # CONDITION 3
        # VALIDATION LOSS
        # ========================================================

        loss_fail = False

        if (
            len(history["val_loss"])
            >= self.min_history
        ):

            previous_loss = (
                history["val_loss"][-1]
            )

            loss_change = (
                self._relative_change(
                    val_loss,
                    previous_loss,
                )
            )

            adaptive_loss_threshold = (
                self._adaptive_threshold(
                    history["val_loss"],
                    self.loss_change_threshold,
                )
            )

            loss_fail = (
                loss_change
                > adaptive_loss_threshold
            )

        else:

            loss_change = 0.0

            adaptive_loss_threshold = (
                self.loss_change_threshold
            )

        # ========================================================
        # CONDITION 4
        # HISTORICAL VALIDATION MSE
        # ========================================================

        mse_history_fail = False

        mean_change = 0.0
        std_change = 0.0

        if (
            len(history["mse_mean"])
            >= self.min_history
        ):

            previous_mean = (
                history["mse_mean"][-1]
            )

            previous_std = (
                history["mse_std"][-1]
            )

            mean_change = (
                self._relative_change(
                    mse_mean,
                    previous_mean,
                )
            )

            std_change = (
                self._relative_change(
                    mse_std,
                    previous_std,
                )
            )

            mse_history_fail = (
                mean_change
                > self.mse_change_threshold
                or
                std_change
                > self.mse_std_change_threshold
            )

        # ========================================================
        # FOUR SECURITY FAILURES
        # ========================================================

        failed_conditions = sum(
            [
                int(magnitude_fail),
                int(timing_fail),
                int(loss_fail),
                int(mse_history_fail),
            ]
        )

        # ========================================================
        # TRUST / RISK
        # ========================================================

        risk_score = (
            failed_conditions / 4.0
        )

        trust_score = (
            1.0 - risk_score
        )

        # ========================================================
        # FINAL ROUTING
        # ========================================================

        # Security severity has priority over latency.

        if failed_conditions >= 3:

            decision = "QUARANTINE"
            validation_required = True

        elif failed_conditions >= 1:

            decision = "SECONDARY_CHECK"
            validation_required = True

        else:

            if latency_fail:

                decision = "SECONDARY_CHECK"
                validation_required = True

            else:

                decision = "DIRECT"
                validation_required = False

        # ========================================================
        # DIAGNOSTICS
        # ========================================================

        diagnostics = {

            "client_id": client_id,

            # Condition 1
            "magnitude": magnitude,
            "magnitude_threshold": (
                magnitude_threshold
            ),
            "Magnitude_Fail": bool(
                magnitude_fail
            ),

            # Condition 2
            "train_time": train_time,
            "dataset_size": dataset_size,
            "train_time_per_sample": (
                time_per_sample
            ),
            "timing_change": timing_change,
            "Timing_Fail": bool(
                timing_fail
            ),

            # Latency routing
            "arrival_latency": (
                arrival_latency
            ),
            "Latency_Fail": bool(
                latency_fail
            ),
            "latency_class": latency_class,

            # Condition 3
            "val_loss": val_loss,
            "loss_change": loss_change,
            "loss_threshold": (
                adaptive_loss_threshold
            ),
            "Loss_Fail": bool(
                loss_fail
            ),

            # Condition 4
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mse_max": mse_max,
            "mse_mean_change": mean_change,
            "mse_std_change": std_change,
            "MSE_History_Fail": bool(
                mse_history_fail
            ),

            # Overall
            "failed_conditions": (
                failed_conditions
            ),
            "risk_score": risk_score,
            "trust_score": trust_score,

            "decision": decision,
            "validation_required": (
                validation_required
            ),
        }

        # ========================================================
        # LOG
        # ========================================================

        logger.info(
            "[SecurityBuffer] Client-%s | "
            "Latency=%.3fs (%s) | "
            "Magnitude_Fail=%s | "
            "Timing_Fail=%s | "
            "Loss_Fail=%s | "
            "MSE_History_Fail=%s | "
            "Failures=%d/4 | "
            "Trust=%.2f | "
            "Decision=%s",
            client_id,
            arrival_latency,
            latency_class,
            magnitude_fail,
            timing_fail,
            loss_fail,
            mse_history_fail,
            failed_conditions,
            trust_score,
            decision,
        )

        return diagnostics

    # ============================================================
    # APPLY ROUTE METADATA
    # ============================================================

    def _prepare_secondary_update(
        self,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:

        item = copy.deepcopy(
            update
        )

        item["route"] = (
            "SECONDARY_CHECK"
        )

        item["validation_required"] = True

        item["weight_factor"] = 0.7

        item["buffer_age"] = int(
            item.get(
                "buffer_age",
                0
            )
        )

        return item

    def _prepare_quarantine_update(
        self,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:

        item = copy.deepcopy(
            update
        )

        item["route"] = (
            "QUARANTINE"
        )

        item["validation_required"] = True

        item["weight_factor"] = 0.3

        item["buffer_age"] = int(
            item.get(
                "buffer_age",
                0
            )
        )

        return item

    # ============================================================
    # COLLECT CURRENT ROUND UPDATES
    # ============================================================

    def collect_current_round_updates(
        self,
        incoming_updates: List[Dict[str, Any]],
        global_model: nn.Module = None,
        val_loader=None,
        criterion=None,
        global_mse=None,
        device=None,
    ) -> List[Dict[str, Any]]:
        """
        Process ONLY new updates arriving in the current round.

        The extra arguments are accepted for compatibility with
        main.py but are intentionally NOT used here.

        Delayed/buffered updates are handled separately by main.py.
        """

        if global_model is not None:

            self.set_global_model(
                global_model
            )

        ready_updates: List[
            Dict[str, Any]
        ] = []

        next_buffer: List[
            Dict[str, Any]
        ] = []

        # ========================================================
        # PROCESS NEW CURRENT-ROUND UPDATES
        # ========================================================

        for update in incoming_updates:

            client_id = str(
                update.get(
                    "client_id",
                    "unknown"
                )
            )

            # ----------------------------------------------------
            # Safety first
            # ----------------------------------------------------

            if not self._is_update_finite(
                update
            ):

                logger.warning(
                    "[SecurityBuffer] "
                    "Client-%s current update "
                    "contains NaN/Inf -> DROP",
                    client_id,
                )

                continue

            # ----------------------------------------------------
            # Evaluate four security conditions
            # + latency classification
            # ----------------------------------------------------

            diagnostics = (
                self.evaluate_update_behavior(
                    update
                )
            )

            update_item = (
                copy.deepcopy(update)
            )

            update_item[
                "security_diagnostics"
            ] = diagnostics

            failed_conditions = (
                diagnostics[
                    "failed_conditions"
                ]
            )

            latency_fail = (
                diagnostics[
                    "Latency_Fail"
                ]
            )

            # ====================================================
            # CASE 1
            # 3-4 SECURITY FAILURES
            # ====================================================

            if failed_conditions >= 3:

                update_item = (
                    self._prepare_quarantine_update(
                        update_item
                    )
                )

                update_item[
                    "buffer_age"
                ] = 0

                next_buffer.append(
                    update_item
                )

                logger.warning(
                    "[SecurityBuffer] "
                    "Client-%s -> QUARANTINE | "
                    "Failures=%d/4 | "
                    "Latency=%.3fs | "
                    "Factor=0.3 | "
                    "Held for NEXT round",
                    client_id,
                    failed_conditions,
                    diagnostics[
                        "arrival_latency"
                    ],
                )

                continue

            # ====================================================
            # CASE 2
            # 1-2 SECURITY FAILURES
            # ====================================================

            if failed_conditions >= 1:

                update_item = (
                    self._prepare_secondary_update(
                        update_item
                    )
                )

                update_item[
                    "buffer_age"
                ] = 0

                next_buffer.append(
                    update_item
                )

                logger.info(
                    "[SecurityBuffer] "
                    "Client-%s -> SECONDARY BUFFER | "
                    "Failures=%d/4 | "
                    "Latency=%.3fs | "
                    "Factor=0.7 | "
                    "Held for NEXT round",
                    client_id,
                    failed_conditions,
                    diagnostics[
                        "arrival_latency"
                    ],
                )

                continue

            # ====================================================
            # CASE 3
            # 0 SECURITY FAILURES + SLOW
            # ====================================================

            if latency_fail:

                update_item = (
                    self._prepare_secondary_update(
                        update_item
                    )
                )

                update_item[
                    "buffer_age"
                ] = 0

                next_buffer.append(
                    update_item
                )

                logger.info(
                    "[SecurityBuffer] "
                    "Client-%s -> SLOW / "
                    "SECONDARY BUFFER | "
                    "Failures=0/4 | "
                    "Latency=%.3fs > %.3fs | "
                    "Factor=0.7 | "
                    "Held for NEXT round",
                    client_id,
                    diagnostics[
                        "arrival_latency"
                    ],
                    self.latency_threshold,
                )

                continue

            # ====================================================
            # CASE 4
            # 0 SECURITY FAILURES + FAST
            # ====================================================

            update_item[
                "route"
            ] = "DIRECT"

            update_item[
                "validation_required"
            ] = False

            update_item[
                "weight_factor"
            ] = 1.0

            update_item[
                "buffer_age"
            ] = 0

            # ONLY FAST + CLEAN updates
            # enter current-round aggregation.
            ready_updates.append(
                update_item
            )

            # Remember clean direct update
            # for future client history.
            self._remember_update(
                update_item
            )

            logger.info(
                "[SecurityBuffer] "
                "Client-%s -> DIRECT | "
                "FAST | "
                "Failures=0/4 | "
                "Latency=%.3fs <= %.3fs | "
                "Factor=1.0 | "
                "CURRENT round aggregation",
                client_id,
                diagnostics[
                    "arrival_latency"
                ],
                self.latency_threshold,
            )

        # ========================================================
        # SAVE ONLY DELAYED UPDATES
        # ========================================================

        self.buffer = next_buffer

        # ========================================================
        # SUMMARY
        # ========================================================

        direct_clients = [
            str(
                item.get(
                    "client_id",
                    "unknown"
                )
            )
            for item in ready_updates
        ]

        buffered_clients = [
            str(
                item.get(
                    "client_id",
                    "unknown"
                )
            )
            for item in self.buffer
        ]

        logger.info(
            "[SecurityBuffer] "
            "CURRENT round direct aggregation: %s",
            direct_clients,
        )

        logger.info(
            "[SecurityBuffer] "
            "NEXT round buffer: %s",
            buffered_clients,
        )

        logger.info(
            "[SecurityBuffer] "
            "Current ready=%d | "
            "Buffered for next round=%d",
            len(ready_updates),
            len(self.buffer),
        )

        # ONLY current-round DIRECT updates
        # are returned.
        return ready_updates

    # ============================================================
    # BUFFER STATUS
    # ============================================================

    def get_buffer_status(self):

        clients = []

        for item in self.buffer:

            clients.append(
                {
                    "client_id": item.get(
                        "client_id",
                        "unknown",
                    ),
                    "route": item.get(
                        "route",
                        "UNKNOWN",
                    ),
                    "age": item.get(
                        "buffer_age",
                        0,
                    ),
                    "weight_factor": item.get(
                        "weight_factor",
                        1.0,
                    ),
                    "validation_required": item.get(
                        "validation_required",
                        False,
                    ),
                }
            )

        return {
            "count": len(
                self.buffer
            ),
            "clients": clients,
            "window_size": (
                self.window_size
            ),
            "latency_threshold": (
                self.latency_threshold
            ),
            "history_clients": list(
                self.client_history.keys()
            ),
        }
