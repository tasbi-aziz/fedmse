"""
Security-centric semi-asynchronous federated anomaly detection.

    python main_secure.py --config Configuration/nbaiot-10clients-noniid.json \
                          --buffer security --attack scale --attack_strength 5

Pipeline of one communication round
-----------------------------------
    1. Updates held back in the previous round are released
       (SECONDARY weight 0.7, validated QUARANTINE weight 0.3).
    2. Every client trains the VAE locally on its own normal traffic.
    3. Malicious clients (if any) manipulate their update.
    4. Security Buffer routes every update:
           NaN/Inf            -> DROP
           FAST + clean       -> DIRECT     (this round,  weight 1.0)
           SLOW + clean       -> SECONDARY  (next round,  weight 0.7)
           any failed check   -> QUARANTINE -> server-side validation
                                   passed -> next round, weight 0.3
                                   failed -> DROP
    5. The accepted updates are aggregated into the global model.
    6. The global model is evaluated (validation MSE, AUC, F1).

--buffer off       : no buffer, every update is aggregated at once (baseline)
--buffer latency   : delay-only buffer, no security checks      (baseline)
--buffer security  : the proposed Security Buffer
"""

import os
import sys
import json
import copy
import time
import random
import argparse
import logging

# Comet must be imported before torch. It is only used with --comet.
USE_COMET = "--comet" in sys.argv
if USE_COMET:
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        sys.exit("--comet needs the comet_ml package:  pip install comet_ml")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from DataLoader.dataloader import load_data, IoTDataset, IoTDataProcessor
from Trainer import ClientTrainer, GlobalAggregator
from Trainer.security_buffer import (
    SecurityBuffer, validate_on_server, DIRECT, SECONDARY, QUARANTINE, DROP
)
from Trainer.malicious_update import manipulate_update
from Model.AutoEncoder import Autoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main_secure")


# ================================================================
# COMET LOGGING (optional, switched on with --comet)
# ================================================================
# Uses the existing Evaluator/comet_logger.py, so the graphs appear in the same
# Comet project ("security-buffer") with the same metric names as before.
# A Comet problem never stops the training run.

def _comet_logger():
    """Loads Evaluator/comet_logger.py directly (without the rest of Evaluator)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluator", "comet_logger.py")
    spec = importlib.util.spec_from_file_location("comet_logger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comet_start(args, run_idx, seed, n_clients, n_features, malicious):
    if not USE_COMET:
        return None
    try:
        exp = _comet_logger().create_experiment(
            model_type=args.tag, run_number=run_idx + 1, run_seed=seed,
            num_rounds=args.rounds, epoch=args.epochs, learning_rate=args.lr,
            shrink_dim=args.latent, batch_size=args.batch_size, server_lr=args.server_lr,
            update_type=args.server_mode, network_size=n_clients, raw_features=n_features,
        )
        exp.log_parameters({
            "buffer": args.buffer, "attack": args.attack,
            "attack_strength": args.attack_strength if args.attack != "none" else 0,
            "malicious_clients": ",".join(malicious), "attack_start_round": args.attack_start_round,
            "latency_threshold": args.latency_threshold, "rejection_ratio": args.rejection_ratio,
            "hidden": args.hidden, "kl_weight": args.kl_weight,
        })
        return exp
    except Exception as exc:
        log.warning("Comet logging is off for this run: %s", exc)
        return None


def comet_round(exp, rec, extra):
    if exp is None:
        return
    try:
        _comet_logger().log_round_metrics(
            experiment=exp, round_number=rec["round"], global_val_mse=rec["global_val_mse"],
            train_total_loss=extra["train_total_loss"], validation_total_loss=rec["mean_local_val_mse"],
            train_reconstruction_mse=extra["train_reconstruction_mse"],
            validation_reconstruction_mse=rec["mean_local_val_mse"],
            test_precision=rec["test_precision"], test_recall=rec["test_recall"],
            test_f1=rec["test_f1"], test_auc=rec["test_auc"], mean_client_auc=rec["mean_client_auc"],
            direct_updates=rec["direct"], aggregated_updates=rec["aggregated"],
            secondary_updates=rec["secondary"], secondary_accepted=rec["secondary"],
            quarantine_accepted=rec["quarantine_accepted"], dropped_updates=rec["dropped"],
            buffered_updates=extra["waiting_for_next_round"], round_duration=rec["seconds"],
        )
        # extra graphs that the old logger did not have
        exp.log_metrics({
            "quarantined_updates": rec["quarantined"],
            "attacked_updates": extra["attacked"],
            "attacked_updates_quarantined": extra["attacked_quarantined"],
            "attacked_updates_dropped": extra["attacked_dropped"],
            "honest_updates_quarantined": extra["honest_quarantined"],
            "min_client_auc": rec["min_client_auc"],
            "detection_threshold": rec["threshold"],
        }, step=rec["round"])
    except Exception as exc:
        log.warning("Comet round logging failed: %s", exc)


def comet_finish(exp):
    if exp is None:
        return
    try:
        _comet_logger().finish_experiment(exp)
    except Exception as exc:
        log.warning("Comet finish failed: %s", exc)


# ================================================================
# DEFAULT SETTINGS (every one of them can be changed from the command line)
# ================================================================

DEFAULTS = dict(
    rounds=10,                 # communication rounds
    epochs=15,                 # local epochs per round
    lr=1e-3,                   # local learning rate
    batch_size=64,
    runs=3,                    # independent repetitions (different seeds)
    features=64,               # features kept by variance-based selection (115 = keep all)
    log_transform=1,           # 1 = clip negatives to 0 and apply log(1+x); 0 = skip (ablation)
    scaler="standard",         # "standard" or "none" (none = raw values, ablation only)
    hidden=32,                 # VAE hidden layer
    latent=16,                 # VAE latent dimension
    kl_weight=1e-4,
    server_mode="fedavg",      # "fedavg" (SGD server step) or "fedopt" (Adam server step)
    server_lr=1.0,
    bootstrap_fraction=0.10,   # share of each client's training data sent to the server once
    latency_threshold=1.5,     # seconds: FAST / SLOW boundary
    secondary_weight=0.7,
    quarantine_weight=0.3,
    rejection_ratio=2.0,       # quarantine: drop if candidate MSE > ratio * global MSE
    attack_start_round=3,      # rounds 1-2 are clean so every client has a history
    data_seed=1234,
)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# EVALUATION HELPERS
# ================================================================

@torch.no_grad()
def reconstruction_errors(model, x, device, batch=4096):
    """Per-sample reconstruction MSE (the anomaly score)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        recon = model(xb)[0]
        out.append(((recon - xb) ** 2).mean(dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def state_mse(model_template, state, x, device):
    """Reconstruction MSE on x of a model with the given weights."""
    m = copy.deepcopy(model_template).to(device)
    m.load_state_dict(state)
    return float(reconstruction_errors(m, x, device).mean())


def detection_metrics(errors, labels, threshold):
    pred = (errors > threshold).astype(int)
    return dict(
        auc=float(roc_auc_score(labels, errors)),
        precision=float(precision_score(labels, pred, zero_division=0)),
        recall=float(recall_score(labels, pred, zero_division=0)),
        f1=float(f1_score(labels, pred, zero_division=0)),
    )


def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def client_drift(updates, global_state):
    """
    Measures how much the client updates of one round disagree (client drift).

    cosine    : mean pairwise cosine similarity between the client updates
                (1 = all clients move in the same direction, 0 = unrelated).
    alignment : || weighted mean update || / weighted mean || update ||
                (1 = no cancellation when averaging, lower = more drift).
    """
    keys = [k for k, v in global_state.items() if v.is_floating_point()]
    g = torch.cat([global_state[k].flatten().float() for k in keys])
    deltas = torch.stack([torch.cat([u["weights"][k].flatten().float() for k in keys]) - g for u in updates])
    n = torch.tensor([float(u["dataset_size"]) for u in updates])
    norms = deltas.norm(dim=1).clamp_min(1e-12)
    unit = deltas / norms[:, None]
    cos = unit @ unit.T
    m = len(updates)
    mean_cos = float((cos.sum() - cos.diag().sum()) / max(m * (m - 1), 1))
    w = n / n.sum()
    alignment = float((w[:, None] * deltas).sum(0).norm() / (w * norms).sum())
    return mean_cos, alignment


# ================================================================
# DATA
# ================================================================

class RawProcessor:
    """No preprocessing at all: the raw feature values are used as they are.
    Only for the ablation that shows why preprocessing is needed."""

    def fit_transform(self, frame):
        return self.transform(frame)

    def transform(self, frame, type="normal"):
        x = frame.values.astype(np.float32)
        label = np.zeros(len(x), dtype=np.float32) if type == "normal" else np.ones(len(x), dtype=np.float32)
        return x, label


_PROBE = {}


def load_clients(config, data_path, args):
    """Loads every client's files and applies the common preprocessing."""
    names = [d["name"] for d in config["devices_list"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Client names in the config are not unique: {names}")

    rng = random.Random(args.data_seed)
    raw = []
    for dev in config["devices_list"]:
        def read(key):
            return (load_data(os.path.join(data_path, dev[key]))
                    .sample(frac=1, random_state=args.data_seed).reset_index(drop=True))

        normal = read("normal_data_path")
        n_train, n_valid = int(0.4 * len(normal)), int(0.1 * len(normal))
        train = normal[:n_train].reset_index(drop=True)
        n_boot = max(1, int(args.bootstrap_fraction * len(train)))
        raw.append(dict(
            name=dev["name"],
            bootstrap=train[:n_boot],
            train=train[n_boot:].reset_index(drop=True),
            valid=normal[n_train:n_train + n_valid].reset_index(drop=True),
            test_normal=read("test_normal_data_path"),
            abnormal=read("abnormal_data_path"),
            # Simulated device behaviour (seconds). Taken from the config when
            # present so that FAST / SLOW clients are reproducible.
            sim_train_time=float(dev.get("simulated_training_time", rng.uniform(1.0, 5.0))),
            sim_comm_time=float(dev.get("simulated_comm_time", rng.uniform(0.2, 2.5))),
        ))

    # Common preprocessing fitted once on the small server bootstrap set:
    # clip negatives to 0 -> log(1+x) -> top-variance features -> StandardScaler
    if args.scaler == "none":
        processor = RawProcessor()
    else:
        processor = IoTDataProcessor(scaler="standard", use_log_transform=bool(args.log_transform),
                                     n_selected_features=args.features)
    boot_x, boot_y = processor.fit_transform(pd.concat([c["bootstrap"] for c in raw], ignore_index=True))

    pin = torch.cuda.is_available()
    clients = []
    # ---- server attack probe set (behaviour check) ----
    probe_parts, probe_take = [], {}
    if getattr(args, "behavior_ratio", -1) > 0 or args.reserve_probe:
        if args.probe_file:
            px, _ = processor.transform(pd.read_csv(args.probe_file, header=None), type="abnormal")
            rs = np.random.default_rng(args.data_seed)
            probe_parts.append(px[rs.permutation(len(px))[:args.probe_n]])
        else:
            honest = [c["name"] for c in raw if c["name"] not in getattr(args, "_malicious", [])]
            for nm in honest:
                probe_take[nm] = max(1, args.probe_n // len(honest))
    for c in raw:
        tr_x, tr_y = processor.transform(c["train"])
        va_x, va_y = processor.transform(c["valid"])
        tn_x, _ = processor.transform(c["test_normal"])
        ab_x, _ = processor.transform(c["abnormal"], type="abnormal")
        if probe_take.get(c["name"], 0) > 0:
            k = probe_take[c["name"]]
            probe_parts.append(ab_x[-k:]); ab_x = ab_x[:-k]     # server probe rows leave the test set
        poison_loader = None
        if args.attack == "poison" and c["name"] in getattr(args, "_malicious", []):
            # DATA POISONING: replace a fraction of the client's normal training
            # data with its own attack traffic (dataset size stays the same).
            # The attack samples used for poisoning are removed from its test set.
            n_p = min(int(args.poison_ratio * len(tr_x)), len(ab_x) // 2)
            p_x = np.vstack([tr_x[:len(tr_x) - n_p], ab_x[:n_p]]).astype(tr_x.dtype)
            p_y = np.zeros(len(p_x), dtype=tr_y.dtype)
            poison_loader = DataLoader(IoTDataset(p_x, p_y), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
            ab_x = ab_x[n_p:]
            log.info("  %s poisoned: %d of %d training rows replaced by attack traffic", c["name"], n_p, len(tr_x))
        clients.append(dict(
            poison_loader=poison_loader,
            name=c["name"],
            train_loader=DataLoader(IoTDataset(tr_x, tr_y), batch_size=args.batch_size, shuffle=True, pin_memory=pin),
            valid_loader=DataLoader(IoTDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False, pin_memory=pin),
            valid_x=torch.tensor(va_x),
            test_x=torch.tensor(np.vstack([tn_x, ab_x])),
            test_y=np.r_[np.zeros(len(tn_x)), np.ones(len(ab_x))],
            n_train=len(tr_x),
            sim_train_time=c["sim_train_time"],
            sim_comm_time=c["sim_comm_time"],
        ))
    _PROBE["x"] = torch.tensor(np.vstack(probe_parts).astype(np.float32)) if probe_parts else None
    if _PROBE["x"] is not None:
        log.info("Server attack probe set: %d rows (%s)", len(_PROBE["x"]), "file" if args.probe_file else "honest clients' attack data")
    bootstrap_loader = DataLoader(IoTDataset(boot_x, boot_y), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    return clients, bootstrap_loader, boot_x.shape[1]


# ================================================================
# ONE RUN
# ================================================================


# ================================================================
# BASELINE ROBUST AGGREGATORS (for comparison with existing methods)
# ================================================================

def robust_aggregate(method, global_state, updates, n_malicious_assumed, server_delta=None, trim=0.2):
    """Return a new global state from client updates with an existing robust rule.
    median  : coordinate-wise median (Yin et al., 2018)
    trimmed : coordinate-wise trimmed mean, trim fraction per side (Yin et al., 2018)
    krum    : Krum, select the single most central update (Blanchard et al., 2017)
    fltrust : FLTrust, cosine trust vs a server update on root data (Cao et al., 2021)
    """
    keys = [k for k in global_state if global_state[k].is_floating_point()]
    shapes = [global_state[k].shape for k in keys]
    sizes = [global_state[k].numel() for k in keys]
    g = torch.cat([global_state[k].flatten().float() for k in keys])
    D = torch.stack([torch.cat([u["weights"][k].flatten().float() for k in keys]) - g for u in updates])
    D = torch.nan_to_num(D, nan=0.0, posinf=1e6, neginf=-1e6)
    n = D.shape[0]
    if method == "median":
        agg = D.median(dim=0).values
    elif method == "trimmed":
        b = int(trim * n)
        srt = D.sort(dim=0).values
        agg = srt[b:n - b].mean(dim=0) if n - 2 * b > 0 else srt.mean(dim=0)
    elif method == "krum":
        f = n_malicious_assumed
        dist = torch.cdist(D, D) ** 2
        m = max(1, n - f - 2)
        scores = torch.stack([dist[i][torch.arange(n) != i].sort().values[:m].sum() for i in range(n)])
        agg = D[int(scores.argmin())]
    elif method == "fltrust":
        s0 = server_delta
        s0n = s0.norm().clamp_min(1e-12)
        dn = D.norm(dim=1).clamp_min(1e-12)
        ts = torch.relu((D @ s0) / (dn * s0n))
        if ts.sum() <= 0:
            agg = torch.zeros_like(g)
        else:
            agg = ((ts / ts.sum())[:, None] * (D * (s0n / dn)[:, None])).sum(0)
    else:
        raise ValueError(method)
    new = dict(global_state)
    off = 0
    for k, shp, sz in zip(keys, shapes, sizes):
        new[k] = (g[off:off + sz] + agg[off:off + sz]).view(shp).to(global_state[k].dtype)
        off += sz
    return new

def run_once(run_idx, clients, bootstrap_loader, n_features, args, device, malicious):
    seed = (run_idx + 1) * 10000
    set_seeds(seed)
    jitter = np.random.default_rng(seed)          # simulated latency / time noise

    server_val_x = torch.cat([c["valid_x"] for c in clients])
    all_test_x = torch.cat([c["test_x"] for c in clients])
    all_test_y = np.concatenate([c["test_y"] for c in clients])

    # ---------------- initial global model ----------------
    global_model = Autoencoder(input_dim=n_features, hidden_neus=args.hidden,
                               latent_dim=args.latent, output_dim=n_features).to(device)
    init = ClientTrainer(model=global_model, client_id="SERVER_INIT", epoch=1, lr_rate=args.lr,
                         update_type="fedavg", device=str(device),
                         save_dir=os.path.join(args.output_dir, "server_initial"), kl_weight=args.kl_weight)
    init.run(bootstrap_loader, None)
    global_model.load_state_dict(init.get_parameters())

    if args.server_mode == "fedavg":
        aggregator = GlobalAggregator(model=global_model, update_type="fedavg", server_lr=args.server_lr,
                                      max_server_update_norm=None, max_server_step_norm=None)
    else:
        aggregator = GlobalAggregator(model=global_model, update_type="fedopt", server_lr=args.server_lr)

    buffer = None
    if args.buffer != "off":
        buffer = SecurityBuffer(latency_threshold=args.latency_threshold,
                                security_checks=(args.buffer == "security"),
                                cos_min=(args.cos_min if args.cos_min > -1.5 else None),
                                behavior_ratio=(args.behavior_ratio if args.behavior_ratio > 0 else None))
    probe_ax = _PROBE.get("x")
    probe_nx = server_val_x[:len(probe_ax)] if probe_ax is not None else None

    experiment = comet_start(args, run_idx, seed, len(clients), n_features, malicious)

    carryover = []            # updates waiting for the NEXT round: dict(client, delta, n, weight, route)
    round_rows, routing_rows = [], []

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        global_state = cpu_state(aggregator.model.state_dict())
        cost = dict(sec=0.0, fwd=0, bwd=0)
        _t = time.perf_counter()
        global_mse = state_mse(global_model, global_state, server_val_x, device)
        if args.buffer == "security":
            cost["sec"] += time.perf_counter() - _t; cost["fwd"] += len(server_val_x)

        to_aggregate = []

        # ---- 1. release updates held back in the previous round ----
        # A delayed update was trained from the PREVIOUS global model, so its
        # change (delta) is re-applied on top of the CURRENT global model.
        for item in carryover:
            weights = {k: (global_state[k] + item["delta"][k]) if k in item["delta"] else global_state[k]
                       for k in global_state}
            to_aggregate.append(dict(client_id=item["client"], weights=weights,
                                     dataset_size=item["n"], weight_factor=item["weight"]))
        released = len(carryover)
        carryover = []

        # ---- 2. local training + 3. attack ----
        counts = {DIRECT: 0, SECONDARY: 0, QUARANTINE: 0, DROP: 0, "quarantine_accepted": 0}
        local_val, local_train_total, local_train_recon = [], [], []
        incoming = []
        for c in clients:
            trainer = ClientTrainer(model=aggregator.model, client_id=c["name"],
                                    save_dir=os.path.join(args.output_dir, "clients", c["name"]),
                                    epoch=args.epochs, lr_rate=args.lr, update_type="fedavg",
                                    device=str(device), kl_weight=args.kl_weight)
            use_poison = (c.get("poison_loader") is not None and rnd >= args.attack_start_round)
            trainer.run(c["poison_loader"] if use_poison else c["train_loader"], c["valid_loader"])
            local_val.append(trainer.val_loss)
            local_train_recon.append(trainer.reconstruction_loss)
            local_train_total.append(trainer.reconstruction_loss + args.kl_weight * trainer.kl_loss)

            update = dict(
                client_id=c["name"],
                weights=cpu_state(trainer.get_parameters()),
                dataset_size=trainer.dataset_size,
                val_loss=trainer.val_loss,
                val_mse_list=list(trainer.val_mse_list),
                # simulated device timing (+-10 % noise every round)
                train_time=c["sim_train_time"] * (1 + jitter.uniform(-0.1, 0.1)),
                arrival_time=c["sim_comm_time"] * (1 + jitter.uniform(-0.1, 0.1)),
            )

            is_malicious = c["name"] in malicious and rnd >= args.attack_start_round and args.attack != "none"
            if is_malicious:
                if args.attack != "poison":
                    update = manipulate_update(update, global_state, args.attack, args.attack_strength)

            incoming.append((c, update, bool(is_malicious)))

        honest = [u for _, u, mal in incoming if not mal]
        drift_cos, drift_align = client_drift(honest, global_state)

        # ---- 4. routing of all updates received in this round's window ----
        _t = time.perf_counter()
        if buffer is not None:
            bscores = None
            if buffer.behavior_ratio is not None and buffer.security_checks and probe_ax is not None:
                bscores = {}
                for _, u, _ in incoming:
                    try:
                        ea = state_mse(global_model, u["weights"], probe_ax, device)
                        en = state_mse(global_model, u["weights"], probe_nx, device)
                        sc = ea / max(en, 1e-12)
                    except Exception:
                        sc = float("nan")
                    bscores[str(u["client_id"])] = sc if np.isfinite(sc) else float("nan")
                    cost["fwd"] += len(probe_ax) + len(probe_nx)
            buffer.begin_round([u for _, u, _ in incoming], global_state, behavior_scores=bscores)
        cost["sec"] += time.perf_counter() - _t

        afl_reject = set()
        _t = time.perf_counter()
        if args.aflguard > 0:
            cost["fwd"] += args.epochs * len(bootstrap_loader.dataset)
            cost["bwd"] += args.epochs * len(bootstrap_loader.dataset)
            # AFLGuard (Fang et al., ACSAC 2022): server update on trusted root data,
            # accept a client update only if ||g_i - g_s|| <= lambda * ||g_s||
            st = ClientTrainer(model=aggregator.model, client_id="SERVER_ROOT",
                               save_dir=os.path.join(args.output_dir, "server_root"),
                               epoch=args.epochs, lr_rate=args.lr, update_type="fedavg",
                               device=str(device), kl_weight=args.kl_weight)
            st.run(bootstrap_loader, None)
            sw = cpu_state(st.get_parameters())
            kf = [k for k in global_state if global_state[k].is_floating_point()]
            gs = torch.cat([(sw[k] - global_state[k]).flatten().float() for k in kf])
            for c, update, _ in incoming:
                gi = torch.cat([(update["weights"][k] - global_state[k]).flatten().float() for k in kf])
                gi = torch.nan_to_num(gi, nan=1e6, posinf=1e6, neginf=-1e6)
                if (gi - gs).norm() > args.aflguard * gs.norm():
                    afl_reject.add(c["name"])

        cost["sec"] += time.perf_counter() - _t
        for c, update, is_malicious in incoming:
            _t = time.perf_counter()
            row = dict(run=run_idx + 1, round=rnd, client=c["name"], attacked=is_malicious,
                       latency=round(update["arrival_time"], 3))
            if c["name"] in afl_reject:
                counts[DROP] += 1
                row.update(latency_class="-", failed_checks="aflguard", warmup=False,
                           route=DROP, outcome="dropped", weight=0.0)
                routing_rows.append(row)
                continue

            if buffer is None:                                   # no buffer at all
                route, outcome, weight = DIRECT, "aggregated_now", 1.0
                row.update(latency_class="-", failed_checks="", warmup=False)
            else:
                diag = buffer.route(update, global_state)
                route = diag["route"]
                row.update(latency_class=diag["latency_class"], failed_checks="+".join(diag["failed_checks"]),
                           warmup=diag["warmup"], relative_magnitude=diag.get("measured", {}).get("magnitude"), cosine=diag.get("cosine"),
                           behavior_score=diag.get("behavior_score"), behavior_median=diag.get("behavior_median"),
                           raw_magnitude=diag.get("raw_magnitude"))
                if route == DIRECT:
                    outcome, weight = "aggregated_now", 1.0
                elif route == SECONDARY:
                    outcome, weight = "next_round", args.secondary_weight
                elif route == QUARANTINE and "behavior" in diag["failed_checks"]:
                    # the probe test IS the server-side validation for this failure:
                    # a model that has learned attack traffic as normal is never admitted
                    outcome, weight = "dropped", 0.0
                elif route == QUARANTINE:
                    # server-side validation of the suspicious local model
                    cost["fwd"] += len(server_val_x)
                    try:
                        cand_mse = state_mse(global_model, update["weights"], server_val_x, device)
                    except Exception:
                        cand_mse = float("inf")
                    row.update(candidate_mse=cand_mse, global_mse=global_mse)
                    if validate_on_server(cand_mse, global_mse, args.rejection_ratio):
                        outcome, weight = "next_round", args.quarantine_weight
                        counts["quarantine_accepted"] += 1
                    else:
                        outcome, weight = "dropped", 0.0
                else:
                    outcome, weight = "dropped", 0.0

            cost["sec"] += time.perf_counter() - _t
            counts[route] += 1
            row.update(route=route, outcome=outcome, weight=weight)
            routing_rows.append(row)

            if outcome == "aggregated_now":
                to_aggregate.append(dict(client_id=c["name"], weights=update["weights"],
                                         dataset_size=update["dataset_size"], weight_factor=1.0))
            elif outcome == "next_round":
                delta = {k: update["weights"][k] - global_state[k]
                         for k in global_state if global_state[k].is_floating_point()}
                carryover.append(dict(client=c["name"], delta=delta, n=update["dataset_size"],
                                      weight=weight, route=route))

        # ---- 5. aggregation ----
        if to_aggregate and args.robust != "none":
            server_delta = None
            if args.robust == "fltrust":
                st = ClientTrainer(model=aggregator.model, client_id="SERVER_ROOT",
                                   save_dir=os.path.join(args.output_dir, "server_root"),
                                   epoch=args.epochs, lr_rate=args.lr, update_type="fedavg",
                                   device=str(device), kl_weight=args.kl_weight)
                st.run(bootstrap_loader, None)
                sw = cpu_state(st.get_parameters())
                keys_f = [k for k in global_state if global_state[k].is_floating_point()]
                server_delta = torch.cat([(sw[k] - global_state[k]).flatten().float() for k in keys_f])
            n_mal = max(1, int(round(0.2 * len(to_aggregate))))
            new_state = robust_aggregate(args.robust, global_state, to_aggregate, n_mal, server_delta)
            aggregator.model.load_state_dict({k: v.to(next(aggregator.model.parameters()).device) for k, v in new_state.items()})
        elif to_aggregate:
            aggregator.aggregate(client_updates=to_aggregate)

        # ---- 6. evaluation of the new global model ----
        val_err = reconstruction_errors(aggregator.model, server_val_x, device)
        # detection threshold = 95th percentile of NORMAL validation errors
        # (never taken from the test set)
        threshold = float(np.percentile(val_err, 95))
        test_err = reconstruction_errors(aggregator.model, all_test_x, device)
        if not np.all(np.isfinite(test_err)):
            test_err = np.nan_to_num(test_err, nan=1e30, posinf=1e30)
        g = detection_metrics(test_err, all_test_y, threshold)
        client_auc = {}
        for c in clients:
            e = np.nan_to_num(reconstruction_errors(aggregator.model, c["test_x"], device), nan=1e30, posinf=1e30)
            client_auc[c["name"]] = float(roc_auc_score(c["test_y"], e))

        rec = dict(run=run_idx + 1, round=rnd, global_val_mse=float(np.mean(val_err)),
                   mean_local_val_mse=float(np.mean(local_val)), threshold=threshold,
                   update_cosine=drift_cos, update_alignment=drift_align,
                   test_auc=g["auc"], test_f1=g["f1"], test_precision=g["precision"], test_recall=g["recall"],
                   mean_client_auc=float(np.mean(list(client_auc.values()))),
                   min_client_auc=float(np.min(list(client_auc.values()))),
                   aggregated=len(to_aggregate), released_from_previous=released,
                   direct=counts[DIRECT], secondary=counts[SECONDARY], quarantined=counts[QUARANTINE],
                   quarantine_accepted=counts["quarantine_accepted"],
                   dropped=sum(1 for r in routing_rows if r["run"] == run_idx + 1 and r["round"] == rnd and r["outcome"] == "dropped"),
                   seconds=round(time.time() - t0, 2), client_auc=client_auc,
                   defense_seconds=cost["sec"], server_fwd_samples=cost["fwd"], server_bwd_samples=cost["bwd"])
        round_rows.append(rec)

        this_round = [r for r in routing_rows if r["run"] == run_idx + 1 and r["round"] == rnd]
        comet_round(experiment, rec, dict(
            train_total_loss=float(np.mean(local_train_total)),
            train_reconstruction_mse=float(np.mean(local_train_recon)),
            waiting_for_next_round=len(carryover),
            attacked=sum(r["attacked"] for r in this_round),
            attacked_quarantined=sum(r["attacked"] and r["route"] == QUARANTINE for r in this_round),
            attacked_dropped=sum(r["attacked"] and r["outcome"] == "dropped" for r in this_round),
            honest_quarantined=sum((not r["attacked"]) and r["route"] == QUARANTINE for r in this_round),
        ))
        log.info("[Run %d Round %d/%d] ValMSE %.4f | AUC %.4f | F1 %.4f | MeanClientAUC %.4f | "
                 "direct %d secondary %d quarantine %d (accepted %d) dropped %d | aggregated %d",
                 run_idx + 1, rnd, args.rounds, rec["global_val_mse"], rec["test_auc"], rec["test_f1"],
                 rec["mean_client_auc"], rec["direct"], rec["secondary"], rec["quarantined"],
                 rec["quarantine_accepted"], rec["dropped"], rec["aggregated"])

    comet_finish(experiment)
    torch.save(aggregator.model.state_dict(), os.path.join(args.output_dir, f"{args.tag}_run{run_idx + 1}.pth"))
    return round_rows, routing_rows


# ================================================================
# SUMMARY
# ================================================================

def summarise(round_rows, routing_rows, args, malicious):
    rounds = pd.DataFrame([{k: v for k, v in r.items() if k != "client_auc"} for r in round_rows])
    final = rounds[rounds["round"] == args.rounds]
    best = rounds.loc[rounds.groupby("run")["test_auc"].idxmax()]

    def ms(s):
        return dict(mean=float(s.mean()), std=float(s.std(ddof=0)))

    summary = dict(
        final_round=dict(test_auc=ms(final["test_auc"]), test_f1=ms(final["test_f1"]),
                         mean_client_auc=ms(final["mean_client_auc"]), global_val_mse=ms(final["global_val_mse"])),
        best_round_auc=ms(best["test_auc"]),
        worst_val_mse_any_round=float(rounds["global_val_mse"].max()),
    )

    routing = pd.DataFrame(routing_rows)
    if args.buffer != "off" and len(routing):
        checked = routing[(routing["round"] >= args.attack_start_round)]
        mal = checked[checked["attacked"]]
        hon = checked[~checked["attacked"]]
        summary["security_buffer"] = dict(
            malicious_updates=int(len(mal)),
            malicious_quarantined=int((mal["route"] == QUARANTINE).sum()),
            malicious_dropped=int((mal["outcome"] == "dropped").sum()),
            malicious_accepted_reduced_weight=int(((mal["route"] == QUARANTINE) & (mal["outcome"] == "next_round")).sum()),
            malicious_missed=int((mal["route"].isin([DIRECT, SECONDARY])).sum()),
            honest_updates=int(len(hon)),
            honest_quarantined=int((hon["route"] == QUARANTINE).sum()),
            honest_dropped=int((hon["outcome"] == "dropped").sum()),
        )
    return summary, rounds, routing


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Security Buffer federated anomaly detection")
    p.add_argument("--config", required=True, help="client configuration JSON")
    p.add_argument("--data_path", default=None, help="overrides data_path of the config")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--tag", default=None, help="name used for the output files")
    p.add_argument("--buffer", choices=["off", "latency", "security"], default="security")
    p.add_argument("--attack", choices=["none", "scale", "noise", "weight_scale", "report", "time", "loss", "mse", "poison"], default="none")
    p.add_argument("--poison_ratio", type=float, default=0.5, help="data poisoning: fraction of training rows replaced by attack traffic")
    p.add_argument("--attack_strength", type=float, default=5.0)
    p.add_argument("--malicious", default="Client-7,Client-8", help="comma separated client names")
    p.add_argument("--robust", choices=["none", "median", "trimmed", "krum", "fltrust"], default="none",
                   help="existing robust aggregation rule (use with --buffer off)")
    p.add_argument("--behavior_ratio", type=float, default=-1.0, help="behaviour check: fail if probe score < ratio * round median (<=0 disables)")
    p.add_argument("--reserve_probe", type=int, default=1, help="always hold out the probe rows from the test set (same test set with/without the check)")
    p.add_argument("--probe_n", type=int, default=300, help="size of the server attack probe set")
    p.add_argument("--probe_file", default=None, help="headerless CSV of attack rows for the server probe set (else taken from honest clients' attack data)")
    p.add_argument("--cos_min", type=float, default=-2.0, help="direction check: min cosine to the round median update (-2 disables)")
    p.add_argument("--aflguard", type=float, default=0.0, help="AFLGuard lambda (>0 enables; use with --buffer latency)")
    p.add_argument("--comet", action="store_true", help="also log every round to Comet (graphs)")
    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(value), default=value)
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    data_path = args.data_path or config["data_path"]
    malicious = [m.strip() for m in args.malicious.split(",") if m.strip()] if args.attack != "none" else []
    names = [d["name"] for d in config["devices_list"]]
    unknown = [m for m in malicious if m not in names]
    if unknown:
        raise ValueError(f"Malicious clients {unknown} are not in the config. Client names are: {names}")

    if args.tag is None:
        args.tag = f"{args.buffer}_{args.attack}" + (f"{args.attack_strength:g}" if args.attack != "none" else "")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.data_seed)

    args._malicious = malicious
    clients, bootstrap_loader, n_features = load_clients(config, data_path, args)
    log.info("Clients: %d | features: %d | buffer: %s | attack: %s (strength %s) on %s from round %d",
             len(clients), n_features, args.buffer, args.attack, args.attack_strength, malicious, args.attack_start_round)
    for c in clients:
        log.info("  %-10s train rows %5d | simulated latency %.2fs (%s)", c["name"], c["n_train"], c["sim_comm_time"],
                 "FAST" if c["sim_comm_time"] <= args.latency_threshold else "SLOW")

    all_rounds, all_routing = [], []
    for run_idx in range(args.runs):
        r, q = run_once(run_idx, clients, bootstrap_loader, n_features, args, device, malicious)
        all_rounds += r
        all_routing += q

    summary, rounds_df, routing_df = summarise(all_rounds, all_routing, args, malicious)
    rounds_df.to_csv(os.path.join(args.output_dir, f"{args.tag}_rounds.csv"), index=False)
    routing_df.to_csv(os.path.join(args.output_dir, f"{args.tag}_routing.csv"), index=False)
    with open(os.path.join(args.output_dir, f"{args.tag}_results.json"), "w") as f:
        json.dump(dict(settings=vars(args), malicious_clients=malicious, summary=summary,
                       rounds=all_rounds), f, indent=2)

    log.info("==================== SUMMARY (%s) ====================", args.tag)
    log.info(json.dumps(summary, indent=2))
