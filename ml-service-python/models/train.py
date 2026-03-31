from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pandas is not installed for the active interpreter. "
        "Install it with `python -m pip install pandas`."
    ) from exc

DATASET_PATH = Path(__file__).resolve().parents[1] / "clean_dataset.csv"

data = pd.read_csv(DATASET_PATH)
print(data.head())

features = ["users", "api_instances", "db_connections", "cache_enabled"]

X = data[features]
y_latency = data["latency"]
y_failure = data["failure"]

# Sanity checks before modeling
print(X.isnull().sum())
print(X.describe())
print(y_failure.value_counts())
print(y_latency.describe())

# ── Imports ───────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import numpy as np

# ── 1. Split ──────────────────────────────────────────────
X_train, X_test, y_lat_train, y_lat_test, y_fail_train, y_fail_test = train_test_split(
    X, y_latency, y_failure, test_size=0.2, random_state=42
)

# ── 2. Scale for XGBoost (RF doesn't need this) ───────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # never fit on test data

# ══════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════

rf_latency = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf_latency.fit(X_train, y_lat_train)

rf_failure = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf_failure.fit(X_train, y_fail_train)

# Evaluate RF
y_lat_pred_rf  = rf_latency.predict(X_test)
y_fail_pred_rf = rf_failure.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_lat_test, y_lat_pred_rf))

print(f"\nRF Latency RMSE : {rf_rmse:.4f}")
print(classification_report(y_fail_test, y_fail_pred_rf))

for name, model in [("Latency", rf_latency), ("Failure", rf_failure)]:
    print(f"\n{name} Feature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:20s} {imp:.4f}")

# ══════════════════════════════════════════════════════════
# XGBOOST  (uses scaled features)
# ══════════════════════════════════════════════════════════

xgb_latency = XGBRegressor(
    n_estimators=800,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=70
)
xgb_latency.fit(
    X_train_scaled, y_lat_train,
    eval_set=[(X_test_scaled, y_lat_test)],
    verbose=False
)

xgb_failure = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=677/323,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)
xgb_failure.fit(
    X_train_scaled, y_fail_train,
    eval_set=[(X_test_scaled, y_fail_test)],
    verbose=False
)

# Evaluate XGBoost
y_lat_pred_xgb  = xgb_latency.predict(X_test_scaled)
y_fail_pred_xgb = xgb_failure.predict(X_test_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_lat_test, y_lat_pred_xgb))

best_iter = getattr(xgb_latency, 'best_iteration', xgb_latency.n_estimators)
print(f"\nXGB Latency RMSE : {xgb_rmse:.4f}")
print(f"Best iteration   : {best_iter}")
print(classification_report(y_fail_test, y_fail_pred_xgb))

# ══════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════

print(f"\nRF  RMSE : {rf_rmse:.4f}")
print(f"XGB RMSE : {xgb_rmse:.4f}")
print(f"Winner   : {'XGBoost' if xgb_rmse < rf_rmse else 'Random Forest'}")

import joblib

# Save models
joblib.dump(rf_latency,  "rf_latency.pkl")
joblib.dump(rf_failure,  "rf_failure.pkl")
joblib.dump(xgb_latency, "xgb_latency.pkl")
joblib.dump(xgb_failure, "xgb_failure.pkl")

# Save scaler too — critical for XGBoost inference
joblib.dump(scaler, "scaler.pkl")

print("Models saved.")

import joblib

# Load
rf_latency  = joblib.load("rf_latency.pkl")
xgb_latency = joblib.load("xgb_latency.pkl")
scaler      = joblib.load("scaler.pkl")

# Predict
new_data = [[50000, 5, 100, 1]]   # users, api_instances, db_connections, cache_enabled

# RF — raw features
rf_pred = rf_latency.predict(new_data)

# XGBoost — must scale first
xgb_pred = xgb_latency.predict(scaler.transform(new_data))

print(f"RF  predicted latency : {rf_pred[0]:.2f}ms")
print(f"XGB predicted latency : {xgb_pred[0]:.2f}ms")