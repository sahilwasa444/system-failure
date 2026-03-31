import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load dataset
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset.csv"

data = pd.read_csv(DATASET_PATH)

features = [
    "users",
    "api_instances",
    "db_connections",
    "cache_enabled",
    "cpu_usage",
    "memory_usage",
    "latency"
]

X = data[features].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)


# Autoencoder Model
class AutoEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


model = AutoEncoder(X_tensor.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training
epochs = 100

for epoch in range(epochs):

    output = model(X_tensor)

    loss = criterion(output, X_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")


# Reconstruction error
with torch.no_grad():

    reconstructed = model(X_tensor)

    error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)


threshold = torch.mean(error) + 2 * torch.std(error)

anomaly = error > threshold

data["anomaly"] = anomaly.numpy()


print("\nDetected anomalies:\n")

print(data[data["anomaly"] == True])


# Remove anomalies
clean_data = data[data["anomaly"] == False]

SAVE_PATH = Path(__file__).resolve().parents[1] / "clean_dataset.csv"

clean_data.to_csv(SAVE_PATH, index=False)

print("\nClean dataset saved:", SAVE_PATH)