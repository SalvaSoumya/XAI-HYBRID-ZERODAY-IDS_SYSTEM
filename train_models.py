from src.preprocessing import load_and_preprocess
from src.model import train_isolation_forest, train_xgboost
from src.hybrid import hybrid_predict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

X, y = load_and_preprocess("C:/Users/Hp/Desktop/XAI-ZERODAY-IDS/data/raw/UNSW_NB15_training-set.csv")

# Split once globally
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
iso_model, iso_scaler, iso_metrics = train_isolation_forest(X, y)
xgb_model, xgb_metrics = train_xgboost(X, y)

# Hybrid prediction
hybrid_preds = hybrid_predict(xgb_model, iso_model, iso_scaler, X_test)

hybrid_metrics = {
    "accuracy": accuracy_score(y_test, hybrid_preds),
    "precision": precision_score(y_test, hybrid_preds),
    "recall": recall_score(y_test, hybrid_preds),
    "f1_score": f1_score(y_test, hybrid_preds)
}

print("\nXGBoost:", xgb_metrics)
print("Isolation Forest:", iso_metrics)
print("Hybrid Model:", hybrid_metrics)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, hybrid_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Hybrid Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/hybrid_confusion_matrix.png")
plt.show()