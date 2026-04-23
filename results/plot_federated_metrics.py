import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics_log.csv")


# --------------------------
# Accuracy vs Rounds
# --------------------------

plt.figure()
plt.plot(df["round"], df["accuracy"])
plt.title("Federated Accuracy Across Rounds")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("results/round_accuracy.png")
plt.show()


# --------------------------
# F1 Score
# --------------------------

plt.figure()
plt.plot(df["round"], df["f1_score"])
plt.title("F1 Score Across Rounds")
plt.xlabel("Communication Round")
plt.ylabel("F1 Score")
plt.grid(True)

plt.savefig("results/f1_rounds.png")
plt.show()


# --------------------------
# Precision
# --------------------------

plt.figure()
plt.plot(df["round"], df["precision"])
plt.title("Precision Across Rounds")
plt.xlabel("Communication Round")
plt.ylabel("Precision")
plt.grid(True)

plt.savefig("results/precision_rounds.png")
plt.show()


# --------------------------
# Recall
# --------------------------

plt.figure()
plt.plot(df["round"], df["recall"])
plt.title("Recall Across Rounds")
plt.xlabel("Communication Round")
plt.ylabel("Recall")
plt.grid(True)

plt.savefig("results/recall_rounds.png")
plt.show()


# --------------------------
# Combined comparison graph
# --------------------------

plt.figure()

plt.plot(df["round"], df["accuracy"], label="Accuracy")
plt.plot(df["round"], df["precision"], label="Precision")
plt.plot(df["round"], df["recall"], label="Recall")
plt.plot(df["round"], df["f1_score"], label="F1")

plt.title("Federated Metrics Comparison")
plt.xlabel("Communication Round")
plt.ylabel("Metric Value")

plt.legend()
plt.grid(True)

plt.savefig("results/all_metrics_comparison.png")
plt.show()