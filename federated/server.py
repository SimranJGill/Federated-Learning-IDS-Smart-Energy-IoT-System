import sys
import os
import flwr as fl
import csv
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOG_FILE = "results/metrics_log.csv"

if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "round",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ])

current_round = 0


def log_metrics(metrics):

    global current_round

    current_round += 1

    if not metrics:
        return {}

    aggregated = {}

    keys = metrics[0][1].keys()

    for key in keys:

        values = [m[1][key] for m in metrics]

        aggregated[key] = float(np.median(values))

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            current_round,
            aggregated.get("accuracy", 0),
            aggregated.get("precision", 0),
            aggregated.get("recall", 0),
            aggregated.get("f1_score", 0),
            aggregated.get("roc_auc", 0)
        ])

    print(f"Logged metrics for round {current_round}")

    return aggregated


strategy = fl.server.strategy.FedAvg(

    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,

    evaluate_metrics_aggregation_fn = log_metrics
)


print("Starting Federated Server...")

fl.server.start_server(

    server_address="0.0.0.0:8080",

    strategy=strategy,

    config=fl.server.ServerConfig(num_rounds=20)
)