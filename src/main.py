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
import json
import copy
import time
import random
import argparse
import logging

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
# DEFAULT SETTINGS (every one of them can be changed from the command line)
# ================================================================

DEFAULTS = dict(
    rounds=10,                 # communication rounds
    epochs=15,                 # local epochs per round
    lr=1e-3,                   # local learning rate
    batch_size=64,
    runs=3,                    # independent repetitions (different seeds)
    features=64,               # features kept by variance-based selection
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


# ================================================================
# DATA
# ================================================================

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
    processor = IoTDataProcessor(scaler="standard", use_log_transform=True,
                                 n_selected_features=args.features)
    boot_x, boot_y = processor.fit_transform(pd.concat([c["bootstrap"] for c in raw], ignore_index=True))

    pin = torch.cuda.is_available()
    clients = []
    for c in raw:
        tr_x, tr_y = processor.transform(c["train"])
        va_x, va_y = processor.transform(c["valid"])
        tn_x, _ = processor.transform(c["test_normal"])
        ab_x, _ = processor.transform(c["abnormal"], type="abnormal")
        clients.append(dict(
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
    bootstrap_loader = DataLoader(IoTDataset(boot_x, boot_y), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    return clients, bootstrap_loader, boot_x.shape[1]


# ================================================================
# ONE RUN
# ================================================================

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
                                security_checks=(args.buffer == "security"))

    carryover = []            # updates waiting for the NEXT round: dict(client, delta, n, weight, route)
    round_rows, routing_rows = [], []

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        global_state = cpu_state(aggregator.model.state_dict())
        global_mse = state_mse(global_model, global_state, server_val_x, device)

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
        local_val = []
        incoming = []
        for c in clients:
            trainer = ClientTrainer(model=aggregator.model, client_id=c["name"],
                                    save_dir=os.path.join(args.output_dir, "clients", c["name"]),
                                    epoch=args.epochs, lr_rate=args.lr, update_type="fedavg",
                                    device=str(device), kl_weight=args.kl_weight)
            trainer.run(c["train_loader"], c["valid_loader"])
            local_val.append(trainer.val_loss)

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
                update = manipulate_update(update, global_state, args.attack, args.attack_strength)

            incoming.append((c, update, bool(is_malicious)))

        # ---- 4. routing of all updates received in this round's window ----
        if buffer is not None:
            buffer.begin_round([u for _, u, _ in incoming], global_state)

        for c, update, is_malicious in incoming:
            row = dict(run=run_idx + 1, round=rnd, client=c["name"], attacked=is_malicious,
                       latency=round(update["arrival_time"], 3))

            if buffer is None:                                   # no buffer at all
                route, outcome, weight = DIRECT, "aggregated_now", 1.0
                row.update(latency_class="-", failed_checks="", warmup=False)
            else:
                diag = buffer.route(update, global_state)
                route = diag["route"]
                row.update(latency_class=diag["latency_class"], failed_checks="+".join(diag["failed_checks"]),
                           warmup=diag["warmup"], relative_magnitude=diag.get("measured", {}).get("magnitude"),
                           raw_magnitude=diag.get("raw_magnitude"))
                if route == DIRECT:
                    outcome, weight = "aggregated_now", 1.0
                elif route == SECONDARY:
                    outcome, weight = "next_round", args.secondary_weight
                elif route == QUARANTINE:
                    # server-side validation of the suspicious local model
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
        if to_aggregate:
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
                   test_auc=g["auc"], test_f1=g["f1"], test_precision=g["precision"], test_recall=g["recall"],
                   mean_client_auc=float(np.mean(list(client_auc.values()))),
                   min_client_auc=float(np.min(list(client_auc.values()))),
                   aggregated=len(to_aggregate), released_from_previous=released,
                   direct=counts[DIRECT], secondary=counts[SECONDARY], quarantined=counts[QUARANTINE],
                   quarantine_accepted=counts["quarantine_accepted"],
                   dropped=sum(1 for r in routing_rows if r["run"] == run_idx + 1 and r["round"] == rnd and r["outcome"] == "dropped"),
                   seconds=round(time.time() - t0, 2), client_auc=client_auc)
        round_rows.append(rec)
        log.info("[Run %d Round %d/%d] ValMSE %.4f | AUC %.4f | F1 %.4f | MeanClientAUC %.4f | "
                 "direct %d secondary %d quarantine %d (accepted %d) dropped %d | aggregated %d",
                 run_idx + 1, rnd, args.rounds, rec["global_val_mse"], rec["test_auc"], rec["test_f1"],
                 rec["mean_client_auc"], rec["direct"], rec["secondary"], rec["quarantined"],
                 rec["quarantine_accepted"], rec["dropped"], rec["aggregated"])

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
    p.add_argument("--attack", choices=["none", "scale", "noise", "weight_scale", "report"], default="none")
    p.add_argument("--attack_strength", type=float, default=5.0)
    p.add_argument("--malicious", default="Client-7,Client-8", help="comma separated client names")
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
