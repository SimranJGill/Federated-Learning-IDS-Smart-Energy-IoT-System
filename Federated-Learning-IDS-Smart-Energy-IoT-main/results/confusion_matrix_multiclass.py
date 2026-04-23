import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import IDSModel


# -----------------------------
# Load test data
# -----------------------------

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)

# For Conv1D input
X_test = X_test.unsqueeze(1)


# -----------------------------
# Load trained model
# -----------------------------

input_size = X_test.shape[2]
num_classes = 5

model = IDSModel(input_size, num_classes)

model.load_state_dict(
    torch.load("models/ids_model.pth", map_location=torch.device("cpu"))
)

model.eval()


# -----------------------------
# Predict
# -----------------------------

with torch.no_grad():

    outputs = model(X_test)

    preds = torch.argmax(outputs, dim=1).numpy()


# -----------------------------
# Confusion Matrix
# -----------------------------

labels = [
    "DoS",
    "Normal",
    "Probe",
    "R2L",
    "U2R"
]

cm = confusion_matrix(y_test, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(8,6))

disp.plot(ax=ax)

plt.title("Multi-Class Confusion Matrix")

plt.savefig("results/confusion_matrix_multiclass.png")

plt.show()