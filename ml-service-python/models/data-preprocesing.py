from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).resolve().parents[1] / "collector" / "traffic_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "latency", "status", "failure"]).copy()

print(df.head())
print(df["timestamp"].min(), df["timestamp"].max())

df = df.sort_values("timestamp").reset_index(drop=True)
print(df.head())

features = ["latency", "status", "failure"]
data = df[features].astype(float)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


def sequences(data, seq_len):
    X = []
    y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    return np.array(X), np.array(y)


seq_len = 10
X, y = sequences(data_scaled, seq_len)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
