# %%
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(".."))
from src.preprocessing import *


# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# %%
train_path = "C:/Users/Hp/Desktop/XAI-ZERODAY-IDS/data/raw/UNSW_NB15_training-set.csv"
test_path = "C:/Users/Hp/Desktop/XAI-ZERODAY-IDS/data/raw/UNSW_NB15_testing-set.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

print(full_df["attack_cat"].unique())

# %%
train_attacks = ['Normal', 'DoS', 'Reconnaissance', 'Generic']
test_attacks = ['Exploits', 'Shellcode', 'Worms', 'Backdoor', 'Analysis', 'Fuzzers']

train_zero_df = full_df[full_df["attack_cat"].isin(train_attacks)].copy()
test_zero_df = full_df[full_df["attack_cat"].isin(test_attacks)].copy()

print("Train Zero:", train_zero_df.shape)
print("Test Zero:", test_zero_df.shape)


# %%
def encode_categorical(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

train_zero_df = encode_categorical(train_zero_df)
test_zero_df = encode_categorical(test_zero_df)

# %%
print(full_df["attack_cat"].unique())
print(full_df["attack_cat"].value_counts())

# %%
X_train = train_zero_df.drop(columns=["label", "attack_cat"])
y_train = train_zero_df["label"]

X_test = test_zero_df.drop(columns=["label", "attack_cat"])
y_test = test_zero_df["label"]

print(X_train.shape)

# %%


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)

print("XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("Precision:", precision_score(y_test, xgb_preds))
print("Recall:", recall_score(y_test, xgb_preds))
print("F1:", f1_score(y_test, xgb_preds))

# %%
X_train_normal = train_zero_df[train_zero_df["label"] == 0]
X_train_normal = X_train_normal.drop(columns=["label", "attack_cat"])

X_train_normal_scaled = scaler.fit_transform(X_train_normal)

iso_model = IsolationForest(
    n_estimators=300,
    contamination=0.2,
    random_state=42
)

iso_model.fit(X_train_normal_scaled)

iso_scores = iso_model.decision_function(X_test_scaled)
iso_preds = np.where(iso_scores < 0, 1, 0)

print("Isolation Forest Results:")
print("Accuracy:", accuracy_score(y_test, iso_preds))
print("Precision:", precision_score(y_test, iso_preds))
print("Recall:", recall_score(y_test, iso_preds))
print("F1:", f1_score(y_test, iso_preds))
print(X_train_normal.shape)

# %%
hybrid_preds = np.where(
    (xgb_preds == 1) | (iso_preds == 1),
    1,
    0
)

print("Hybrid Model Results:")
print("Accuracy:", accuracy_score(y_test, hybrid_preds))
print("Precision:", precision_score(y_test, hybrid_preds))
print("Recall:", recall_score(y_test, hybrid_preds))
print("F1:", f1_score(y_test, hybrid_preds))

# %%
print("Normal training samples:", X_train_normal.shape)
print("Test samples:", X_test.shape)
# Fit scaler only on normal training
scaler = StandardScaler()

X_train_normal_scaled = scaler.fit_transform(X_train_normal)

# Transform test using SAME scaler
X_test_scaled = scaler.transform(X_test)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lime.lime_tabular import LimeTabularExplainer


# %%
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)


# %%
print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("Feature count:", len(X_train.columns))


# %%
# Background dataset (for baseline expectation)
background_size = 100
background = X_train_scaled[
    np.random.choice(X_train_scaled.shape[0], background_size, replace=False)
]


# %%
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.2,
    random_state=42
)
iso_forest.fit(X_train_scaled)
explainer = shap.KernelExplainer(
    iso_forest.decision_function,
    background
)


# %%
samples_to_explain = 50

shap_values = explainer.shap_values(
    X_test_scaled[:samples_to_explain],
    nsamples=100
)


# %%
shap.summary_plot(
    shap_values,
    X_test_scaled[:samples_to_explain],
    feature_names=X_train.columns,
    plot_type="bar"
)


# %%
shap.summary_plot(
    shap_values,
    X_test_scaled[:samples_to_explain],
    feature_names=X_train.columns
)


# %%
index = 0  # any suspicious traffic instance

shap.force_plot(
    explainer.expected_value,
    shap_values[index],
    X_test_scaled[index],
    feature_names=X_train.columns,
    matplotlib=True
)


# %%
shap.dependence_plot(
    "sbytes",
    shap_values,
    X_test_scaled[:samples_to_explain],
    feature_names=X_train.columns
)


# %%



from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X_train.columns,
    class_names=["Normal", "Attack"],
    discretize_continuous=True,
    mode="classification"
)


# %%
def iso_predict_proba(X):
    scores = iso_forest.decision_function(X)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return np.vstack([1 - scores, scores]).T


# %%
from IPython.display import HTML
i = 75 # choose any test sample

lime_exp = lime_explainer.explain_instance(
    X_test_scaled[i],
    iso_predict_proba,
    num_features=75
)

HTML(lime_exp.as_html())


# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb

plt.style.use("seaborn-v0_8")

# %%
df = pd.read_csv( "C:/Users/Hp/Desktop/XAI-ZERODAY-IDS/data/raw/UNSW_NB15_training-set.csv")

df = df.drop(columns=["id", "attack_cat"], errors="ignore")

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
scaler_xgb = StandardScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train)
X_test_xgb = scaler_xgb.transform(X_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)

xgb_model.fit(X_train_xgb, y_train)

# %%
X_train_normal = X_train[y_train == 0]

scaler_iso = StandardScaler()
X_train_normal_scaled = scaler_iso.fit_transform(X_train_normal)
X_test_iso_scaled = scaler_iso.transform(X_test)

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

iso_model.fit(X_train_normal_scaled)

# %%
X_train_normal = X_train[y_train == 0]

scaler_iso = StandardScaler()
X_train_normal_scaled = scaler_iso.fit_transform(X_train_normal)
X_test_iso_scaled = scaler_iso.transform(X_test)

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

iso_model.fit(X_train_normal_scaled)

# %%
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test_xgb)

shap.summary_plot(
    shap_values_xgb,
    X_test,
    plot_type="bar"
)

# %%
shap.summary_plot(
    shap_values_xgb,
    X_test
)

# %%
sample_index = 10

shap.force_plot(
    explainer_xgb.expected_value,
    shap_values_xgb[sample_index],
    X_test.iloc[sample_index],
    matplotlib=True
)

# %%
anomaly_scores = iso_model.decision_function(X_test_iso_scaled)

plt.figure()
sns.histplot(anomaly_scores, bins=50)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()

# %%
plt.figure()
sns.boxplot(x=y_test, y=anomaly_scores)
plt.title("Anomaly Scores vs Actual Label")
plt.xlabel("Actual Label (0=Normal,1=Attack)")
plt.ylabel("Anomaly Score")
plt.show()

# %%
def iso_predict(data):
    return iso_model.decision_function(scaler_iso.transform(data))

explainer_iso = shap.KernelExplainer(
    iso_predict,
    X_train.sample(100)
)

shap_values_iso = explainer_iso.shap_values(
    X_test.sample(50)
)

# %%
shap.summary_plot(
    shap_values_iso,
    X_test.sample(50)
)

# %%
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns,
    class_names=["Normal", "Attack"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    X_test.iloc[5].values,
    xgb_model.predict_proba
)

from IPython.display import display, HTML

html = exp.as_html()
display(HTML(html))

# %%
def iso_predict_proba(data):
    scores = iso_model.decision_function(scaler_iso.transform(data))
    probs = 1 / (1 + np.exp(-scores))
    return np.vstack([1 - probs, probs]).T

exp_iso = lime_explainer.explain_instance(
    X_test.iloc[5].values,
    iso_predict_proba
)

from IPython.display import display, HTML

html = exp.as_html()
display(HTML(html))

# %%
xgb_importance = xgb_model.feature_importances_

plt.figure()
plt.barh(X.columns, xgb_importance)
plt.title("XGBoost Feature Importance")
plt.show()


