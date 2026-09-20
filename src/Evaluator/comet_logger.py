# comet_logger.py

import comet_ml


def create_experiment(
    model_type,
    run_number,
    run_seed,
    num_rounds,
    epoch,
    learning_rate,
    shrink_dim,
    batch_size,
    latency_threshold,
    server_lr,
    update_type,
    network_size,
    raw_features,
    timing_attack_client,
    timing_attack_start_round,
):
    """
    Create and configure a Comet experiment for one model/run.
    """

    experiment = comet_ml.Experiment(
        project_name="security-buffer"
    )

    experiment.set_name(
        f"{model_type}_run_{run_number}"
    )

    experiment.log_parameters({
        "model_type": model_type,
        "run": run_number,
        "seed": run_seed,

        "num_rounds": num_rounds,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "shrink_dim": shrink_dim,
        "batch_size": batch_size,

        "latency_threshold": latency_threshold,
        "server_lr": server_lr,
        "update_type": update_type,

        "network_size": network_size,
        "raw_features": raw_features,

        "timing_attack_client": timing_attack_client,
        "timing_attack_start_round": timing_attack_start_round,
    })

    return experiment


def log_round_metrics(
    experiment,
    round_number,
    global_val_mse,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
    mean_client_auc,
    direct_updates,
    aggregated_updates,
    secondary_updates,
    secondary_accepted,
    quarantine_accepted,
    dropped_updates,
    buffered_updates,
    round_duration,
):
    """
    Log one FL round's metrics to Comet.
    """

    experiment.log_metrics(
        {
            "global_val_mse": float(global_val_mse),

            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc),

            "mean_client_auc": float(mean_client_auc),

            "direct_updates": int(direct_updates),
            "aggregated_updates": int(aggregated_updates),

            "secondary_updates": int(secondary_updates),
            "secondary_accepted": int(secondary_accepted),
            "quarantine_accepted": int(quarantine_accepted),

            "dropped_updates": int(dropped_updates),
            "buffered_updates": int(buffered_updates),

            "round_duration": float(round_duration),
        },
        step=round_number,
    )


def finish_experiment(experiment):
    """
    Finish the Comet experiment safely.
    """

    if experiment is not None:
        experiment.end()
