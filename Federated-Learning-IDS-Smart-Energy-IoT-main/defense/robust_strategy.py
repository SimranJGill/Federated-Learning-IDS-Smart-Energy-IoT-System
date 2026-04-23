import numpy as np

def detect_byzantine_clients(updates, threshold=5.0):
    valid_updates = []
    for update in updates:
        distance = np.linalg.norm(update)

        if distance < threshold:
            valid_updates.append(update)
        else:
            print("⚠️ Malicious client update detected and removed")
    return valid_updates

def median_aggregation(metrics):
    """
    Median aggregation function for Flower server.
    This aggregates client metrics using median instead of average.
    """

    if not metrics:
        return {}

    aggregated = {}

    # metrics format: [(num_examples, metrics_dict), ...]
    keys = metrics[0][1].keys()

    for key in keys:
        values = [m[1][key] for m in metrics]

        # detect malicious updates
        values = detect_byzantine_clients(values)

        if len(values) == 0:
            continue

        aggregated[key] = float(np.median(values))

    return aggregated

def trimmed_mean_aggregation(metrics, trim_ratio=0.2):

    if not metrics:
        return {}

    aggregated = {}

    keys = metrics[0][1].keys()

    for key in keys:

        values = sorted([m[1][key] for m in metrics])

        trim = int(len(values) * trim_ratio)

        trimmed_values = values[trim: len(values) - trim]

        aggregated[key] = float(np.mean(trimmed_values))

    return aggregated

def krum_aggregation(metrics):

    if not metrics:
        return {}

    aggregated = {}

    keys = metrics[0][1].keys()

    for key in keys:

        values = np.array([m[1][key] for m in metrics])

        distances = []

        for i in range(len(values)):

            dist = np.sum((values[i] - values) ** 2)

            distances.append(dist)

        krum_index = np.argmin(distances)

        aggregated[key] = float(values[krum_index])

    return aggregated

