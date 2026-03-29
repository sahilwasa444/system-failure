from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).resolve().parents[1] / "collector" / "traffic_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
df["status"] = pd.to_numeric(df["status"], errors="coerce")
df["failure"] = pd.to_numeric(df["failure"], errors="coerce")

filled_failed_latencies = (df["failure"].eq(1) & df["latency"].isna()).sum()
df.loc[df["failure"].eq(1) & df["latency"].isna(), "latency"] = 0.0

print(f"Filled failed rows with latency=0: {filled_failed_latencies}")

df = df.dropna(subset=["timestamp", "latency", "status", "failure"]).copy()

print(df.head())
print(df["timestamp"].min(), df["timestamp"].max())

df = df.sort_values("timestamp").reset_index(drop=True)
print(df.head())

df["time_delta_seconds"] = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
df["time_delta_seconds"] = df["time_delta_seconds"].clip(lower=0.0)

df["url_encoded"] = df["url"].astype("category").cat.codes
features = [
    "latency",
    "status",
    "failure",
    "url_encoded",
    "time_delta_seconds",
]

target_index = features.index("latency")


def sequences(data, seq_len, target_index):
    X = []
    y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, target_index])   # only next latency

    return np.array(X), np.array(y)

seq_len = 15
split_index = int(len(df) * 0.8)

train_data = df.iloc[:split_index][features].astype(float).to_numpy()
test_data = df.iloc[split_index - seq_len:][features].astype(float).to_numpy()

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train, y_train = sequences(train_scaled, seq_len, target_index)
X_test, y_test = sequences(test_scaled, seq_len, target_index)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
