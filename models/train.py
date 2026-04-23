import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from model import IDSModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Load data
# =========================

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")


# =========================
# Convert tensors
# =========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# Conv1D channel dimension
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)


train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


# =========================
# Model
# =========================

input_size = X_train.shape[2]

num_classes = 5

model = IDSModel(
    input_size,
    num_classes
).to(device)


criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 10

best_acc = 0


print("Starting training...")


for epoch in range(epochs):

    model.train()

    total_loss = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    # =====================
    # Validation
    # =====================

    model.eval()

    with torch.no_grad():

        outputs = model(
            X_test.to(device)
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        acc = (
            preds ==
            y_test.to(device)
        ).float().mean().item()


    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {total_loss:.4f}, "
        f"Val Acc: {acc:.4f}"
    )


    # Save best model only
    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "models/ids_model.pth"
        )


print(
    f"Training complete. "
    f"Best accuracy: {best_acc:.4f}"
)