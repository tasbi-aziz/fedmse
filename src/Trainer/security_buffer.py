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
        cos_sim_threshold: float = -0.1,  # kept only for backward compatibility
        magnitude_threshold: float = 1.0,
        loss_change_threshold: float = 0.50,
        mse_change_threshold: float = 0.50,
        mse_std_change_threshold: float = 0.50,
        history_size: int = 5,
        min_history: int = 2
    ):
        """
        Security Buffer for asynchronous federated learning.

        Current decision signals:
        1. Update magnitude
        2. Workload-aware training time
        3. Local loss behavior
        4. Historical validation-MSE behavior

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

        # 10-second latency threshold as requested
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

        self.history_size = history_size
        self.min_history = min_history

        # ---------------------------------------------------------
        # Delayed updates
        # ---------------------------------------------------------
        self.buffer: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Per-client historical behavior
        #
        # Used to distinguish:
        # legitimate Non-IID behavior
        # from
        # unusual behavior of the same client.
        # ---------------------------------------------------------
        self.client_history: Dict[str, Dict[str, List[float]]] = {}

    # =============================================================
    # GLOBAL MODEL
    # =============================================================

    def set_global_model(self, global_model: nn.Module):
        """Update internal copy of global model."""

        if global_model is not None:
            self.global_model = copy.deepcopy(global_model)

    # =============================================================
    # CLIENT HISTORY
    # =============================================================

    def _get_client_history(self, client_id):
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

        return self.client_history[client_id]

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
        history = self._get_client_history(client_id)

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

            if not math.isfinite(float(value)):
                continue

            history[key].append(float(value))

            if len(history[key]) > self.history_size:
                history[key].pop(0)

    # =============================================================
    # UPDATE MAGNITUDE
    # =============================================================

    def compute_update_magnitude(
        self,
        model_update: dict,
        global_model: nn.Module
    ) -> float:
        """
        Computes ||local_weights - global_weights||.

        IMPORTANT:
        This is NOT cosine similarity.

        It measures the size of the client update rather than
        its direction.

        Only floating-point tensors are considered.
        """

        if global_model is None or not model_update:
            return 0.0

        global_dict = global_model.state_dict()

        squared_sum = 0.0

        for name, local_param in model_update.items():

            if name not in global_dict:
                continue

            global_param = global_dict[name]

            if not torch.is_floating_point(local_param):
                continue

            if not torch.is_floating_point(global_param):
                continue

            local_cpu = local_param.detach().cpu().float()
            global_cpu = global_param.detach().cpu().float()

            if local_cpu.shape != global_cpu.shape:
                continue

            difference = local_cpu - global_cpu

            squared_sum += torch.sum(
                difference * difference
            ).item()

        return math.sqrt(
            max(squared_sum, 0.0)
        )

    # =============================================================
    # MSE STATISTICS
    # =============================================================

    def compute_mse_statistics(self, val_mse_list: list):
        """
        Converts the 5 validation MSE values into bounded statistics.

        log1p is used so extremely large MSE values do not create
        gigantic security scores.
        """

        if not val_mse_list:
            return {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0
            }

        safe_values = []

        for value in val_mse_list:

            try:
                value = float(value)

                if not math.isfinite(value):
                    continue

                # MSE should not be negative.
                value = max(value, 0.0)

                # Log compression prevents score explosion.
                value = math.log1p(value)

                safe_values.append(value)

            except (TypeError, ValueError):
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
            "mean": float(torch.mean(mse_tensor).item()),
            "std": float(torch.std(
                mse_tensor,
                unbiased=False
            ).item()),
            "max": float(torch.max(mse_tensor).item())
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
        """
        Calculates a client-specific adaptive upper threshold.

        With insufficient history, the configured base threshold
        is used.

        Once enough historical data exists:
            threshold = max(base, mean + factor * std)
        """

        if not history_values:
            return base_threshold

        if len(history_values) < self.min_history:
            return base_threshold

        tensor = torch.tensor(
            history_values,
            dtype=torch.float32
        )

        mean_value = float(
            torch.mean(tensor).item()
        )

        std_value = float(
            torch.std(
                tensor,
                unbiased=False
            ).item()
        )

        adaptive_value = (
            mean_value +
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
        """
        Normalizes training time by local dataset size.

        This prevents a large dataset client from being treated as
        suspicious only because it naturally takes longer to train.
        """

        try:
            train_time = float(train_time)
            dataset_size = int(dataset_size)

            if train_time < 0:
                return 0.0

            if dataset_size <= 0:
                return train_time

            return train_time / dataset_size

        except (TypeError, ValueError):
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
            global_model = self.global_model

        client_id = update.get(
            "client_id",
            "unknown"
        )

        weights = update.get(
            "weights",
            {}
        )

        # ---------------------------------------------------------
        # Basic metadata
        # ---------------------------------------------------------

        latency = update.get(
            "arrival_time",
            update.get("train_time", 0.0)
        )

        try:
            latency = float(latency)
        except (TypeError, ValueError):
            latency = 0.0

        dataset_size = update.get(
            "dataset_size",
            update.get("data_size", 0)
        )

        try:
            dataset_size = int(dataset_size)
        except (TypeError, ValueError):
            dataset_size = 0

        val_loss = update.get(
            "val_loss",
            0.0
        )

        try:
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            val_loss = 0.0

        val_mse_list = update.get(
            "val_mse_list",
            []
        )

        # ---------------------------------------------------------
        # 1. Update magnitude
        # ---------------------------------------------------------

        magnitude = self.compute_update_magnitude(
            weights,
            global_model
        )

        # ---------------------------------------------------------
        # 2. Workload-aware training time
        # ---------------------------------------------------------

        time_per_sample = self.compute_time_per_sample(
            latency,
            dataset_size
        )

        # ---------------------------------------------------------
        # 3. MSE statistics
        # ---------------------------------------------------------

        mse_stats = self.compute_mse_statistics(
            val_mse_list
        )

        history = self._get_client_history(
            client_id
        )

        # =========================================================
        # CONDITION 1: MAGNITUDE
        # =========================================================

        magnitude_threshold = self._adaptive_threshold(
            history["magnitude"],
            self.magnitude_threshold
        )

        magnitude_fail = (
            magnitude > magnitude_threshold
            if history["magnitude"]
            else False
        )

        # =========================================================
        # CONDITION 2: WORKLOAD-AWARE TRAINING TIME
        # =========================================================

        time_threshold = self._adaptive_threshold(
            history["train_time_per_sample"],
            self.latency_threshold
        )

        # IMPORTANT:
        # The 10-sec threshold is also preserved as a direct
        # wall-clock upper bound.
        wall_clock_fail = (
            latency > self.latency_threshold
        )

        time_behavior_fail = (
            time_per_sample > time_threshold
            if history["train_time_per_sample"]
            else False
        )

        latency_fail = (
            wall_clock_fail and time_behavior_fail
        )

        # =========================================================
        # CONDITION 3: LOCAL LOSS
        # =========================================================

        loss_threshold = self._adaptive_threshold(
            history["val_loss"],
            self.loss_change_threshold
        )

        loss_fail = False

        if history["val_loss"] and val_loss > 0:

            previous_loss = history["val_loss"][-1]

            if previous_loss > 0:

                relative_loss_change = (
                    abs(val_loss - previous_loss)
                    / (previous_loss + 1e-8)
                )

                loss_fail = (
                    relative_loss_change >
                    loss_threshold
                )

            else:
                loss_fail = False

        # =========================================================
        # CONDITION 4: MSE HISTORY CHANGE
        # =========================================================

        mse_history_fail = False
        mse_mean_change = 0.0
        mse_std_change = 0.0

        if history["mse_mean"]:

            previous_mean = history["mse_mean"][-1]
            previous_std = history["mse_std"][-1]

            current_mean = mse_stats["mean"]
            current_std = mse_stats["std"]

            if previous_mean > 0:

                mse_mean_change = (
                    abs(current_mean - previous_mean)
                    / (abs(previous_mean) + 1e-8)
                )

            if previous_std > 0:

                mse_std_change = (
                    abs(current_std - previous_std)
                    / (abs(previous_std) + 1e-8)
                )

            mse_history_fail = (
                mse_mean_change >
                self.mse_change_threshold
                or
                mse_std_change >
                self.mse_std_change_threshold
            )

        # =========================================================
        # RISK / TRUST
        # =========================================================
        #
        # DO NOT use raw MSE values here.
        #
        # This keeps the score bounded between 0 and 1.
        # =========================================================

        failed_conditions = sum([
            bool(magnitude_fail),
            bool(latency_fail),
            bool(loss_fail),
            bool(mse_history_fail)
        ])

        total_conditions = 4

        risk_score = (
            failed_conditions /
            total_conditions
        )

        trust_score = 1.0 - risk_score

        # =========================================================
        # DECISION
        # =========================================================

        # No failed conditions -> DIRECT
        if failed_conditions == 0:

            decision = "DIRECT"
            validation_required = False

        # One suspicious signal -> deeper checking
        elif failed_conditions <= 2:

            decision = "SECONDARY_CHECK"
            validation_required = True

        # Many conditions fail -> quarantine
        else:

            decision = "QUARANTINE"
            validation_required = False

        return {
            "client_id": client_id,

            "magnitude": magnitude,
            "magnitude_threshold": magnitude_threshold,

            "latency": latency,
            "dataset_size": dataset_size,
            "time_per_sample": time_per_sample,
            "time_threshold": time_threshold,

            "val_loss": val_loss,
            "loss_threshold": loss_threshold,

            "mse_mean": mse_stats["mean"],
            "mse_std": mse_stats["std"],
            "mse_max": mse_stats["max"],

            "mse_mean_change": mse_mean_change,
            "mse_std_change": mse_std_change,

            "Magnitude_Fail": magnitude_fail,
            "Latency_Fail": latency_fail,
            "Loss_Fail": loss_fail,
            "MSE_History_Fail": mse_history_fail,

            "failed_conditions": failed_conditions,

            "risk_score": risk_score,
            "trust_score": trust_score,

            # Backward-compatible field
            "score": trust_score,

            "decision": decision,
            "validation_required": validation_required
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

        """
        Main asynchronous update routing.

        Fast updates inside the 10-second window are checked.

        Slow updates are placed into secondary checking/buffer.

        IMPORTANT:
        No cosine similarity is used.
        """

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

            client_id = buffered_item.get(
                "client_id",
                "unknown"
            )

            # -----------------------------------------------------
            # Expiration
            # -----------------------------------------------------

            if buffered_item["age"] > self.window_size:

                logging.warning(
                    f"[Security Buffer] DROPPED expired update | "
                    f"Client: {client_id} | "
                    f"Reached Max Age ({self.window_size})"
                )

                continue

            # -----------------------------------------------------
            # Secondary checking
            # -----------------------------------------------------

            behavior = self.evaluate_update_behavior(
                buffered_item,
                self.global_model
            )

            buffered_item.update(
                behavior
            )

            # -----------------------------------------------------
            # Secondary check PASSED
            # -----------------------------------------------------

            if behavior["decision"] == "DIRECT":

                buffered_item["route"] = "RELEASED"
                buffered_item["validation_required"] = False

                ready_updates.append(
                    buffered_item
                )

                self._remember_update(
                    buffered_item
                )

                logging.info(
                    f"[Security Buffer] RELEASED buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

            elif behavior["decision"] == "SECONDARY_CHECK":

                # Hold for another verification round.
                next_buffer.append(
                    buffered_item
                )

                logging.info(
                    f"[Security Buffer] HOLDING buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

            else:

                logging.warning(
                    f"[Security Buffer] QUARANTINE buffered update | "
                    f"Client: {client_id} | "
                    f"Age: {buffered_item['age']} | "
                    f"Failed: {behavior['failed_conditions']}/4"
                )

                buffered_item["route"] = "QUARANTINE"

        # =========================================================
        # STEP 2:
        # PROCESS NEW INCOMING UPDATES
        # =========================================================

        for update in incoming_updates:

            client_id = update.get(
                "client_id",
                "unknown"
            )

            behavior = self.evaluate_update_behavior(
                update,
                self.global_model
            )

            update.update(
                behavior
            )

            latency = behavior["latency"]

            # -----------------------------------------------------
            # FAST + CLEAN
            # -----------------------------------------------------

            if (
                latency <= self.latency_threshold
                and
                behavior["decision"] == "DIRECT"
            ):

                update["route"] = "DIRECT"
                update["validation_required"] = False

                ready_updates.append(
                    update
                )

                self._remember_update(
                    update
                )

                logging.info(
                    f"[Security Buffer] DIRECT ACCEPT Client {client_id} | "
                    f"Latency: {latency:.2f}s | "
                    f"Magnitude: {behavior['magnitude']:.4f} | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # FAST BUT SUSPICIOUS
            # -----------------------------------------------------

            elif (
                latency <= self.latency_threshold
                and
                behavior["decision"] == "SECONDARY_CHECK"
            ):

                update_copy = copy.deepcopy(
                    update
                )

                update_copy["age"] = 0
                update_copy["route"] = "SECONDARY_CHECK"
                update_copy["validation_required"] = True

                next_buffer.append(
                    update_copy
                )

                logging.info(
                    f"[Security Buffer] SECONDARY CHECK Client {client_id} | "
                    f"Latency: {latency:.2f}s | "
                    f"Failed: {behavior['failed_conditions']}/4 | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # SLOW UPDATE
            # -----------------------------------------------------

            elif latency > self.latency_threshold:

                update_copy = copy.deepcopy(
                    update
                )

                update_copy["age"] = 0
                update_copy["route"] = "SECONDARY_CHECK"
                update_copy["validation_required"] = True

                next_buffer.append(
                    update_copy
                )

                logging.info(
                    f"[Security Buffer] BUFFERED SLOW Client {client_id} | "
                    f"Latency: {latency:.2f}s > "
                    f"{self.latency_threshold:.2f}s | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

            # -----------------------------------------------------
            # STRONGLY SUSPICIOUS
            # -----------------------------------------------------

            else:

                update["route"] = "QUARANTINE"
                update["validation_required"] = False

                logging.warning(
                    f"[Security Buffer] QUARANTINE Client {client_id} | "
                    f"Failed: {behavior['failed_conditions']}/4 | "
                    f"Trust: {behavior['trust_score']:.3f}"
                )

        # =========================================================
        # UPDATE BUFFER
        # =========================================================

        self.buffer = next_buffer

        logging.info(
            f"[Security Buffer] Summary -> "
            f"Direct/Released Updates: {len(ready_updates)} | "
            f"Secondary/Buffered: {len(self.buffer)}"
        )

        return ready_updates

    # =============================================================
    # STORE HISTORY
    # =============================================================

    def _remember_update(self, update):

        client_id = update.get(
            "client_id",
            "unknown"
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

    def get_buffer_status(self) -> dict:

        return {
            "buffered_count": len(self.buffer),

            "buffered_clients": [
                item.get("client_id")
                for item in self.buffer
            ],

            "window_size": self.window_size,

            "latency_threshold": self.latency_threshold,

            "history_clients": len(
                self.client_history
            )
        }
