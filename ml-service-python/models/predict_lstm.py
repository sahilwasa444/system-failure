import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("lstm_model.pth")


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# Load dataset
df = pd.read_csv("../collector/traffic_dataset.csv")

df = df.dropna()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["time_delta_seconds"] = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
df["time_delta_seconds"] = df["time_delta_seconds"].clip(lower=0.0)

df["url_encoded"] = df["url"].astype("category").cat.codes


features = [
    "latency",
    "status",
    "failure",
    "url_encoded",
    "time_delta_seconds"
]


data = df[features].values


sequence_length = 15

sequence = data[-sequence_length:]

# Scale the sequence using StandardScaler (same as training)
scaler = StandardScaler()
# Fit on the entire data to match training behavior
scaler.fit(df[features].values)
sequence = scaler.transform(sequence)

sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)


input_size = sequence.shape[2]
hidden_size = 64
output_size = 1


model = LSTMModel(
    input_size,
    hidden_size,
    output_size
)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()


with torch.no_grad():

    prediction = model(sequence)

print("Predicted Next Latency:", prediction.item())