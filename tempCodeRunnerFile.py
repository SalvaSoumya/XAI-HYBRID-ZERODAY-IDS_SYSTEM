from src.preprocessing import load_and_preprocess
from src.model import train_isolation_forest, train_xgboost, save_model
import os
import json

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
X, y = load_and_preprocess("C:/Users/Hp/Desktop/XAI-ZERODAY-IDS/data/raw/UNSW_NB15_training-set.csv")

# Train Isolation Forest
iso_model, iso_metrics = train_isolation_forest(X, y)
save_model(iso_model, "results/isolation_forest.pkl")

# Train XGBoost
xgb_model, xgb_metrics = train_xgboost(X, y)
save_model(xgb_model, "results/xgboost.pkl")

# Save metrics
metrics = {
    "Isolation Forest": iso_metrics,
    "XGBoost": xgb_metrics
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training Completed!")
print(metrics)