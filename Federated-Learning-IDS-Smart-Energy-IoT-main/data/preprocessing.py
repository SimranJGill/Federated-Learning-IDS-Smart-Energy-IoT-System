import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_FILE = "data/raw/smart_energy_iot_attack_data.csv"
PROCESSED_DIR = "data/processed"

print("Starting preprocessing...")

data = pd.read_csv(RAW_FILE)

print("Dataset shape:", data.shape)

# encode categorical columns
data["device_id"] = data["device_id"].astype("category").cat.codes
data["protocol"] = data["protocol"].astype("category").cat.codes

# encode labels
data["label"] = data["label"].apply(lambda x: 0 if x == "Normal" else 1)

# drop timestamp
data = data.drop("timestamp", axis=1)

X = data.drop("label", axis=1)
y = data["label"]

# scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)

np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Preprocessing complete")