
"""
Controlled Malicious Update Experiments

This module creates controlled manipulations of an otherwise
normally trained client update.

IMPORTANT:
Only ONE security condition is manipulated at a time.

Experiments:
    1. Magnitude manipulation
    2. Timing manipulation
    3. Validation-loss manipulation
    4. Validation-MSE manipulation
"""

import copy
import torch


# ================================================================
# EXPERIMENT CONFIGURATION
# ================================================================

# Select ONE experiment at a time.
#
# Options:
#     "none"
#     "magnitude"
#     "timing"
#     "loss"
#     "mse"
#
# Later:
#     "combined"
#
ACTIVE_ATTACK = "combined"


# ================================================================
# 1. MAGNITUDE MANIPULATION
# ================================================================

def manipulate_magnitude(
    update,
    scale_factor=5.0
):
    """
    Artificially increases the magnitude of the client model
    update while keeping the other update metadata unchanged.

    Intended SecurityBuffer signal:
        Magnitude_Fail = True
    """

    manipulated = copy.deepcopy(update)

    original_weights = manipulated["weights"]

    for name, tensor in original_weights.items():

        if torch.is_tensor(tensor):

            manipulated["weights"][name] = (
                tensor * scale_factor
            )

    manipulated["attack_type"] = (
        "magnitude"
    )

    manipulated["attack_parameter"] = (
        scale_factor
    )

    return manipulated


# ================================================================
# 2. TIMING MANIPULATION
# ================================================================

def manipulate_timing(
    update,
    time_factor=3.0
):
    """
    Artificially changes workload-normalized training time.

    Other values remain unchanged.

    Intended SecurityBuffer signal:
        Timing_Fail = True
    """

    manipulated = copy.deepcopy(update)

    original_train_time = float(
        manipulated.get(
            "train_time",
            0.0
        )
    )

    manipulated["train_time"] = (
        original_train_time
        * time_factor
    )

    manipulated["attack_type"] = (
        "timing"
    )

    manipulated["attack_parameter"] = (
        time_factor
    )

    return manipulated


# ================================================================
# 3. VALIDATION LOSS MANIPULATION
# ================================================================

def manipulate_validation_loss(
    update,
    loss_factor=5.0
):
    """
    Artificially changes the reported validation loss.

    Intended SecurityBuffer signal:
        Loss_Fail = True
    """

    manipulated = copy.deepcopy(update)

    original_loss = float(
        manipulated.get(
            "val_loss",
            0.0
        )
    )

    manipulated["val_loss"] = (
        original_loss
        * loss_factor
    )

    manipulated["attack_type"] = (
        "loss"
    )

    manipulated["attack_parameter"] = (
        loss_factor
    )

    return manipulated


# ================================================================
# 4. VALIDATION MSE MANIPULATION
# ================================================================

def manipulate_validation_mse(
    update,
    mse_factor=5.0
):
    """
    Artificially changes the validation MSE history.

    Intended SecurityBuffer signal:
        MSE_History_Fail = True
    """

    manipulated = copy.deepcopy(update)

    original_mse = manipulated.get(
        "val_mse_list",
        []
    )

    manipulated["val_mse_list"] = [

        float(value) * mse_factor

        for value in original_mse
    ]

    manipulated["attack_type"] = (
        "mse"
    )

    manipulated["attack_parameter"] = (
        mse_factor
    )

    return manipulated


# ================================================================
# COMBINED MANIPULATION
# ================================================================

def manipulate_combined(
    update
):
    """
    Manipulates all four security-related signals.

    This experiment is performed only AFTER the four
    individual experiments are tested.
    """

    manipulated = copy.deepcopy(update)

    # ------------------------------------------------------------
    # Magnitude
    # ------------------------------------------------------------

    for name, tensor in manipulated[
        "weights"
    ].items():

        if torch.is_tensor(tensor):

            manipulated["weights"][name] = (
                tensor * 5.0
            )

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    manipulated["train_time"] = (
        float(
            manipulated.get(
                "train_time",
                0.0
            )
        )
        * 3.0
    )

    # ------------------------------------------------------------
    # Validation loss
    # ------------------------------------------------------------

    manipulated["val_loss"] = (
        float(
            manipulated.get(
                "val_loss",
                0.0
            )
        )
        * 5.0
    )

    # ------------------------------------------------------------
    # Validation MSE
    # ------------------------------------------------------------

    manipulated["val_mse_list"] = [

        float(value) * 5.0

        for value in manipulated.get(
            "val_mse_list",
            []
        )
    ]

    manipulated["attack_type"] = (
        "combined"
    )

    manipulated["attack_parameter"] = (
        "magnitude=5x, timing=3x, "
        "loss=5x, mse=5x"
    )

    return manipulated


# ================================================================
# MAIN ATTACK SELECTOR
# ================================================================

def manipulate_update(
    update,
    attack_type=None
):
    """
    Apply the selected controlled manipulation.

    If attack_type is None, ACTIVE_ATTACK is used.
    """

    attack_type = (
        attack_type
        if attack_type is not None
        else ACTIVE_ATTACK
    )

    # ------------------------------------------------------------
    # No attack
    # ------------------------------------------------------------

    if attack_type == "none":

        clean_update = copy.deepcopy(
            update
        )

        clean_update["attack_type"] = (
            "none"
        )

        clean_update["attack_parameter"] = (
            None
        )

        return clean_update

    # ------------------------------------------------------------
    # Magnitude
    # ------------------------------------------------------------

    if attack_type == "magnitude":

        return manipulate_magnitude(
            update,
            scale_factor=5.0
        )

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    if attack_type == "timing":

        return manipulate_timing(
            update,
            time_factor=3.0
        )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    if attack_type == "loss":

        return manipulate_validation_loss(
            update,
            loss_factor=5.0
        )

    # ------------------------------------------------------------
    # MSE
    # ------------------------------------------------------------

    if attack_type == "mse":

        return manipulate_validation_mse(
            update,
            mse_factor=5.0
        )

    # ------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------

    if attack_type == "combined":

        return manipulate_combined(
            update
        )

    raise ValueError(
        f"Unknown attack type: {attack_type}"
    )
