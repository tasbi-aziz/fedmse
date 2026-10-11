"""
Controlled malicious-update experiments.

All model attacks are defined on the UPDATE (delta = W_local - W_global),
which is what a real model-poisoning attacker manipulates:

    scale        W' = W_global + k * delta      (k > 1 : boosted update,
                                                 k < 0 : sign-flipped update)
    noise        W' = W_local + random noise with norm k * ||delta||
    weight_scale W' = k * W_local               (the original crude test)
    report       weights untouched; reported training time x3,
                 validation loss x5, validation-MSE list x5
"""

import copy
import math
import torch


def _is_float(t):
    return torch.is_tensor(t) and t.is_floating_point()


def manipulate_update(update, global_state, attack="none", strength=5.0):
    u = copy.deepcopy(update)
    u["attack_type"] = attack
    u["attack_strength"] = None if attack == "none" else strength
    if attack == "none":
        return u

    w = u["weights"]

    if attack == "scale":
        for k, g in global_state.items():
            if _is_float(g):
                g = g.detach().to(w[k].device)
                w[k] = g + strength * (w[k] - g)

    elif attack == "noise":
        sq = sum(float(torch.sum((w[k] - g.to(w[k].device)) ** 2)) for k, g in global_state.items() if _is_float(g))
        delta_norm = math.sqrt(sq)
        noise = {k: torch.randn_like(w[k]) for k, g in global_state.items() if _is_float(g)}
        noise_norm = math.sqrt(sum(float(torch.sum(n ** 2)) for n in noise.values()))
        for k, n in noise.items():
            w[k] = w[k] + n * (strength * delta_norm / max(noise_norm, 1e-12))

    elif attack == "weight_scale":
        for k in w:
            if _is_float(w[k]):
                w[k] = w[k] * strength

    elif attack == "report":
        u["train_time"] = float(u["train_time"]) * 3.0
        u["val_loss"] = float(u["val_loss"]) * 5.0
        u["val_mse_list"] = [float(v) * 5.0 for v in u["val_mse_list"]]

    else:
        raise ValueError(f"Unknown attack: {attack}")

    return u
