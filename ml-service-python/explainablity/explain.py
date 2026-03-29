import shap
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_failure.pkl"
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
data = pd.read_csv(DATASET_PATH)

features = [
    "users",
    "api_instances",
    "db_connections",
    "cache_enabled"
]

X = data[features]

# Scale features (try this)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# SHAP Explainer - using KernelExplainer as alternative
explainer = shap.KernelExplainer(model.predict, shap.sample(X_scaled, 100))

# Take latest sample
sample = X_scaled.iloc[-1:]

# Get prediction
prediction = model.predict(sample)
prediction_proba = model.predict_proba(sample)

shap_values = explainer.shap_values(sample)

print("\n=== Model Prediction ===")
print(f"Class Prediction: {prediction[0]}")
print(f"Prediction Probability: {prediction_proba[0]}")

print("\n=== Sample Feature Values ===")
for feature, value in zip(features, sample.values[0]):
    print(f"{feature}: {value}")

print("\n=== Dataset Statistics ===")
print(X_scaled.describe())

print("\n=== SHAP Explainability ===")
print(f"Base Score (Expected Value): {explainer.expected_value}")
print(f"SHAP Values (Feature Contributions):\n")

for feature, value in zip(features, shap_values[0]):
    print(f"{feature} : {value:.6f}")

print("\n=== DEBUG: Is Model Working? ===")
# Test if model actually uses features - change one feature
test_sample1 = sample.copy()
test_sample2 = sample.copy()
test_sample2['users'] = test_sample2['users'].max() * 2  # Change users significantly

pred1 = model.predict_proba(test_sample1)[0]
pred2 = model.predict_proba(test_sample2)[0]

print(f"Prediction with original features: {pred1}")
print(f"Prediction with changed users: {pred2}")
print(f"Did prediction change? {not (pred1 == pred2).all()}")

if (pred1 == pred2).all():
    print("❌ PROBLEM: Model ignores feature changes! Model might be broken or overfitted.")