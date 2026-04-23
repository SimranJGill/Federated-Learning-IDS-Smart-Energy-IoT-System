import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Column names (41 features + label + difficulty)
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

#Load dataset
train_df = pd.read_csv("data/KDDTrain+.txt", names=columns)
test_df = pd.read_csv("data/KDDTest+.txt", names=columns)

# Drop difficulty column
train_df.drop("difficulty", axis=1, inplace=True)
test_df.drop("difficulty", axis=1, inplace=True)

#Map labels to 5 classes
def map_label(label):
    if label == "normal":
        return "normal"
    elif label in ["neptune","smurf","back","teardrop","pod","land"]:
        return "dos"
    elif label in ["ipsweep","nmap","portsweep","satan"]:
        return "probe"
    elif label in ["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient"]:
        return "r2l"
    else:
        return "u2r"

train_df["label"] = train_df["label"].apply(map_label)
test_df["label"] = test_df["label"].apply(map_label)

#Encode categorical features
categorical_cols = ["protocol_type", "service", "flag"]

encoder = LabelEncoder()
for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

#Encode labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
test_df["label"] = label_encoder.transform(test_df["label"])

#Split features and labels
X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Save processed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Preprocessing done!")
print("Classes:", label_encoder.classes_)