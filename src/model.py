import joblib
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# 1. Isolation Forest
# -----------------------------
def train_isolation_forest(X, y):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train ONLY on normal traffic
    X_train_normal = X_train[y_train == 0]

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_test = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42
    )

    model.fit(X_train_normal)

    preds = model.predict(X_test)

    preds = [1 if p == -1 else 0 for p in preds]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }

    return model, scaler, metrics


# -----------------------------
# 2. XGBoost (FIXED VERSION)
# -----------------------------
def train_xgboost(X, y):

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale AFTER split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }

    return model, metrics


# -----------------------------
# 3. Save Model
# -----------------------------
def save_model(model, path):
    joblib.dump(model, path)