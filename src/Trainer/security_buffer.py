"""
Security Buffer (semi-asynchronous, security-centric update router).

Design implemented here
-----------------------
Every incoming client update is judged on two independent axes:

    1. Arrival latency   -> FAST (<= latency_threshold) or SLOW
    2. Behaviour         -> CLEAN (0 failed checks) or SUSPICIOUS (>= 1 failed check)

    NaN / Inf             -> DROP
    FAST + CLEAN          -> DIRECT      (aggregated now,        weight 1.0)
    SLOW + CLEAN          -> SECONDARY   (aggregated next round, weight 0.7)
    FAST/SLOW + SUSPICIOUS-> QUARANTINE  (server-side validation decides:
                                          accepted -> next round, weight 0.3
                                          rejected -> DROP)

The four behavioural checks compare the update with the SAME client's own
clean history (median of the last `history_size` clean rounds):

    1. Update magnitude        ||W_local - W_global||_2, measured RELATIVE to the
                               median magnitude of the updates received in the
                               same round, grew too much.  (Honest updates get
                               smaller as training converges, so the raw norm
                               cannot be compared across rounds; the relative
                               size of a client is stable.)
    2. Training time / sample  changed too much (either direction)
    3. Validation loss         rose too much
    4. Validation-MSE folds    worst fold or spread rose too much

Only CLEAN updates (fast OR slow) are written to the history, so a suspicious
update can never become the reference for later rounds.  A client needs
`min_history` clean rounds before it can be checked (warm-up).

The server-side validation of quarantined updates is done by the caller
(main_secure.py) with `validate_on_server`, because it needs the server
validation data.
"""

import math
import logging
from statistics import median
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DIRECT = "DIRECT"
SECONDARY = "SECONDARY"
QUARANTINE = "QUARANTINE"
DROP = "DROP"


class SecurityBuffer:

    def __init__(
        self,
        latency_threshold: float = 1.5,
        magnitude_ratio: float = 2.0,
        time_change: float = 0.5,
        loss_ratio: float = 1.5,
        mse_ratio: float = 1.5,
        mse_std_ratio: float = 3.0,
        history_size: int = 5,
        min_history: int = 2,
        security_checks: bool = True,
    ):
        self.latency_threshold = latency_threshold
        self.magnitude_ratio = magnitude_ratio
        self.time_change = time_change
        self.loss_ratio = loss_ratio
        self.mse_ratio = mse_ratio
        self.mse_std_ratio = mse_std_ratio
        self.history_size = history_size
        self.min_history = min_history
        # security_checks=False gives a conventional "delay-only" buffer
        # (FAST -> now, SLOW -> next round) used as a comparison baseline.
        self.security_checks = security_checks
        self.history: Dict[str, Dict[str, List[float]]] = {}
        self.round_median_magnitude: Optional[float] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_finite(obj: Any) -> bool:
        if torch.is_tensor(obj):
            return bool(torch.isfinite(obj).all().item()) if obj.is_floating_point() else True
        if isinstance(obj, dict):
            return all(SecurityBuffer.is_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(SecurityBuffer.is_finite(v) for v in obj)
        if isinstance(obj, (float, int)):
            return math.isfinite(float(obj))
        return True

    @staticmethod
    def update_magnitude(weights: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> float:
        """L2 norm of (local weights - global weights) over all float tensors."""
        total = 0.0
        for name, g in global_state.items():
            if name in weights and torch.is_tensor(g) and g.is_floating_point():
                d = weights[name].detach().float().cpu() - g.detach().float().cpu()
                total += float(torch.sum(d * d))
        return math.sqrt(total)

    def begin_round(self, updates: List[Dict[str, Any]], global_state: Dict[str, torch.Tensor]) -> float:
        """
        Call once per round with all updates received in this round's window.
        Stores the median update magnitude, the yardstick for check 1.
        The median is not moved by a minority of abnormal updates.
        """
        mags = [self.update_magnitude(u["weights"], global_state) for u in updates
                if self.is_finite(u["weights"])]
        self.round_median_magnitude = float(median(mags)) if mags else None
        return self.round_median_magnitude

    def _hist(self, client_id: str) -> Dict[str, List[float]]:
        return self.history.setdefault(
            client_id,
            {"magnitude": [], "time_per_sample": [], "val_loss": [], "mse_max": [], "mse_std": []},
        )

    def _reference(self, client_id: str, key: str) -> Optional[float]:
        values = self._hist(client_id)[key]
        if len(values) < self.min_history:
            return None
        return float(median(values))

    def remember(self, client_id: str, measured: Dict[str, float]) -> None:
        """Store a CLEAN update's measurements as this client's reference."""
        h = self._hist(client_id)
        for key in h:
            h[key].append(float(measured[key]))
            del h[key][:-self.history_size]

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def route(self, update: Dict[str, Any], global_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Decide the route of ONE incoming update.

        Required keys in `update`:
            client_id, weights, arrival_time, train_time, dataset_size,
            val_loss, val_mse_list
        Returns a diagnostics dict; diagnostics["route"] is one of
        DIRECT / SECONDARY / QUARANTINE / DROP.
        """
        cid = str(update["client_id"])
        latency = float(update["arrival_time"])
        latency_class = "FAST" if latency <= self.latency_threshold else "SLOW"
        diag: Dict[str, Any] = {"client_id": cid, "latency": latency, "latency_class": latency_class}

        check_payload = {k: update[k] for k in ("weights", "arrival_time", "train_time", "val_loss", "val_mse_list")}
        if not self.is_finite(check_payload):
            diag.update(route=DROP, reason="NaN/Inf", failed_checks=[], warmup=False)
            return diag

        mse_list = [float(v) for v in update["val_mse_list"]]
        mse_mean = sum(mse_list) / max(len(mse_list), 1)
        raw_magnitude = self.update_magnitude(update["weights"], global_state)
        yardstick = self.round_median_magnitude or raw_magnitude or 1.0
        diag["raw_magnitude"] = raw_magnitude
        measured = {
            # relative magnitude: 1.0 = typical update of this round
            "magnitude": raw_magnitude / max(yardstick, 1e-12),
            "time_per_sample": float(update["train_time"]) / max(int(update["dataset_size"]), 1),
            "val_loss": float(update["val_loss"]),
            "mse_max": max(mse_list) if mse_list else 0.0,
            "mse_std": math.sqrt(sum((v - mse_mean) ** 2 for v in mse_list) / max(len(mse_list), 1)),
        }
        diag["measured"] = measured

        failed: List[str] = []
        refs = {k: self._reference(cid, k) for k in measured}
        warmup = refs["magnitude"] is None
        diag["reference"] = refs
        diag["warmup"] = warmup

        if self.security_checks and not warmup:
            eps = 1e-12
            # 1. update magnitude grew too much
            if measured["magnitude"] > self.magnitude_ratio * refs["magnitude"] + eps:
                failed.append("magnitude")
            # 2. training time per sample changed too much (either direction)
            rel = abs(measured["time_per_sample"] - refs["time_per_sample"]) / max(refs["time_per_sample"], eps)
            if rel > self.time_change:
                failed.append("timing")
            # 3. validation loss rose too much
            if measured["val_loss"] > self.loss_ratio * refs["val_loss"] + eps:
                failed.append("val_loss")
            # 4. validation-MSE folds: worst fold or spread rose too much
            if (measured["mse_max"] > self.mse_ratio * refs["mse_max"] + eps
                    or measured["mse_std"] > self.mse_std_ratio * refs["mse_std"] + 1e-6):
                failed.append("mse_history")

        diag["failed_checks"] = failed

        if failed:
            diag["route"] = QUARANTINE
        else:
            diag["route"] = DIRECT if latency_class == "FAST" else SECONDARY
            self.remember(cid, measured)          # clean update -> becomes reference

        logger.info(
            "[SecurityBuffer] %s | %.2fs %s | relative magnitude=%.2f | failed=%s%s -> %s",
            cid, latency, latency_class, measured["magnitude"],
            failed if failed else "none", " (warm-up)" if warmup else "", diag["route"],
        )
        return diag


def validate_on_server(candidate_mse: float, global_mse: float, rejection_ratio: float) -> bool:
    """
    Quarantine validation rule.

    The quarantined local model and the current global model are both run on
    the server validation set.  The update is rejected when its reconstruction
    MSE is more than `rejection_ratio` times the global model's MSE.
    """
    if not math.isfinite(candidate_mse):
        return False
    if not math.isfinite(global_mse) or global_mse <= 0:
        return True
    return candidate_mse <= rejection_ratio * global_mse
