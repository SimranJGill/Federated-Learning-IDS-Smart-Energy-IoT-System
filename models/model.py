import torch
import torch.nn as nn
import torch.nn.functional as F


class IDSModel(nn.Module):

    def __init__(self, input_size, num_classes):

        super(IDSModel, self).__init__()

        # =========================
        # CNN BRANCH
        # =========================

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)


        # =========================
        # LSTM BRANCH
        # =========================

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )


        # =========================
        # FEATURE FUSION
        # CNN gives 64 features
        # LSTM gives 64 features
        # Total fusion = 128
        # =========================

        self.fc1 = nn.Linear(128, 128)

        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)


    def forward(self, x):

        # x shape:
        # (batch,1,features)


        # =========================
        # CNN BRANCH
        # =========================

        cnn_x = F.relu(self.conv1(x))

        cnn_x = self.pool(cnn_x)

        cnn_x = F.relu(self.conv2(cnn_x))

        cnn_x = self.pool(cnn_x)


        # Global Average Pooling
        # (batch,64,seq) -> (batch,64)
        cnn_x = self.gap(cnn_x).squeeze(-1)


        # =========================
        # LSTM BRANCH
        # (batch,1,features)
        # -> (batch,features,1)
        # =========================

        lstm_x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(lstm_x)

        # Last hidden state
        lstm_x = lstm_out[:, -1, :]


        # =========================
        # PARALLEL FUSION
        # =========================

        x = torch.cat((cnn_x, lstm_x), dim=1)


        # =========================
        # CLASSIFIER
        # =========================

        x = self.dropout(
            F.relu(
                self.fc1(x)
            )
        )

        # Raw logits for CrossEntropyLoss
        x = self.fc2(x)

        return x