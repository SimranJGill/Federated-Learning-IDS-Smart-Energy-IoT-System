import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import uuid

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import IDSModel


# ==========================
# Load dataset (clean NSL-KDD processed data)
# ==========================

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")


# ==========================
# Non-IID client simulation (NOT poisoning)
# This is data distribution heterogeneity, common in federated learning.
# No labels or samples are altered.
# ==========================

CLIENT_ID = str(uuid.uuid4())[:8]
print("Client ID:", CLIENT_ID)

num_classes = 5

class_indices = {}
for c in range(num_classes):
    class_indices[c] = np.where(y_train == c)[0]

mod = int(CLIENT_ID[-1], 16) % 3

if mod == 0:
    print("Client type: skewed toward classes 0,1")
    selected = np.concatenate([
        class_indices[0][:3000],
        class_indices[1][:3000],
        class_indices[2][:500],
        class_indices[3][:250],
        class_indices[4][:250],
    ])

elif mod == 1:
    print("Client type: skewed toward classes 2,3")
    selected = np.concatenate([
        class_indices[0][:500],
        class_indices[1][:500],
        class_indices[2][:3000],
        class_indices[3][:3000],
        class_indices[4][:250],
    ])

else:
    print("Client type: balanced")
    size = min(len(v) for v in class_indices.values())
    take = min(size, 1000)

    selected = np.concatenate([
        class_indices[c][:take] for c in range(num_classes)
    ])


X_train = X_train[selected]
y_train = y_train[selected]


# ==========================
# Convert to Torch Tensors
# ==========================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Conv1D input
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

dataset = TensorDataset(X_train, y_train)

# Standard mini-batch shuffling (normal practice)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


# ==========================
# Model
# ==========================

input_size = X_train.shape[2]
model = IDSModel(input_size, num_classes)


# ==========================
# Flower Client
# ==========================

class IDSClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_parameters(self, parameters):

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        model.load_state_dict(state_dict, strict=True)


    # ======================
    # Local Training
    # ======================

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()


        return self.get_parameters(config={}), len(loader.dataset), {}


    # ======================
    # Evaluation
    # ======================

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        criterion = torch.nn.CrossEntropyLoss()

        model.eval()

        with torch.no_grad():

            outputs = model(X_test)

            loss = criterion(outputs, y_test).item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true = y_test.cpu().numpy()


        accuracy = accuracy_score(y_true, preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            preds,
            average="weighted",
            zero_division=0
        )

        print("\n===== Evaluation =====")
        print("Loss:", loss)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


        return loss, len(X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": 0.0
        }


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=IDSClient(),
)