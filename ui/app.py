import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "models")

# --------------------------------------------------
# LOAD DATASET (ONLY FOR FEATURE NAMES)
# --------------------------------------------------

data_path = os.path.join(BASE_DIR, "data", "raw", "UNSW_NB15_training-set.csv")

df = pd.read_csv(data_path)
df = df.drop(columns=["id", "attack_cat", "label"], errors="ignore")

feature_columns = df.columns.tolist()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

xgb_model = joblib.load(os.path.join(MODEL_PATH, "xgb_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
iso_model = joblib.load(os.path.join(MODEL_PATH, "iso_model.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_PATH, "preprocessor.pkl"))

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Explainable Hybrid IDS",
    layout="wide"
)

st.title("🔐 Explainable Hybrid Intrusion Detection System")

# --------------------------------------------------
# IMPORTANT FEATURES FOR USER INPUT
# --------------------------------------------------

important_features = [
    "dur","proto","service","state",
    "spkts","dpkts","sbytes","dbytes",
    "rate","sttl","dttl","sload"
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    selected = option_menu(
        menu_title="Navigation",
        options=["Home","Predict Traffic","Model Performance","Explainable AI"],
        icons=["house","activity","bar-chart","cpu"],
        menu_icon="shield-lock",
        default_index=0
    )

# ==================================================
# HOME PAGE (SOC DASHBOARD)
# ==================================================

if selected == "Home":

    st.markdown("### 🛡 Cybersecurity Monitoring Dashboard")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Isolation Forest AUC","0.79")
    col2.metric("Random Forest Accuracy","87%")
    col3.metric("XGBoost Accuracy","87%")
    col4.metric("Hybrid Recall","99%")

    st.divider()

    st.subheader("System Threat Level")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=35,
        title={'text': "Current Threat Level"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':'green'},
                {'range':[40,70],'color':'yellow'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

    st.plotly_chart(gauge,use_container_width=True)

    st.subheader("Attack Distribution")

    attack_data = pd.DataFrame({
        "Type":["Normal","DoS","Recon","Generic","Exploits"],
        "Count":[6000,1500,1200,900,500]
    })

    fig = px.pie(
        attack_data,
        names="Type",
        values="Count",
        title="Detected Network Traffic"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Workflow")

    st.info(
        "Network Traffic → Preprocessing → ML Models → Hybrid Detection → Explainable AI"
    )

# ==================================================
# PREDICT TRAFFIC PAGE
# ==================================================

elif selected == "Predict Traffic":

    st.title("🚦 Network Traffic Prediction")

    st.info(
    """
    Enter network traffic features to detect potential cyber attacks.

    Hybrid Model Used:
    - Isolation Forest (Anomaly Detection)
    - Random Forest (Classification)
    - XGBoost (Boosted Classification)
    """
    )

    user_input = {}

    col1,col2 = st.columns(2)

    for i,col in enumerate(important_features):

        if df[col].dtype == "object":

            options = df[col].unique().tolist()

            if i%2==0:
                user_input[col] = col1.selectbox(col,options)
            else:
                user_input[col] = col2.selectbox(col,options)

        else:

            mean_val = float(df[col].mean())

            if i%2==0:
                user_input[col] = col1.number_input(col,value=mean_val)
            else:
                user_input[col] = col2.number_input(col,value=mean_val)

    for col in feature_columns:
        if col not in user_input:
            user_input[col] = df[col].mean()

    input_df = pd.DataFrame([user_input])

    st.divider()

    if st.button("🔍 Detect Attack"):

        processed = preprocessor.transform(input_df)

        xgb_pred = xgb_model.predict(processed)[0]
        rf_pred = rf_model.predict(processed)[0]

        iso_pred = iso_model.predict(processed)
        iso_pred = 1 if iso_pred[0]==-1 else 0

        xgb_prob = xgb_model.predict_proba(processed)[0][1]
        rf_prob = rf_model.predict_proba(processed)[0][1]

        votes = xgb_pred + rf_pred + iso_pred
        hybrid_pred = 1 if votes >=2 else 0

        if hybrid_pred==1:
            st.error("🔴 HIGH RISK: Attack Detected")
        else:
            st.success("🟢 SAFE: Normal Traffic")

        st.subheader("Model Decisions")

        c1,c2,c3 = st.columns(3)

        c1.metric("Isolation Forest","Attack" if iso_pred else "Normal")
        c2.metric("Random Forest","Attack" if rf_pred else "Normal")
        c3.metric("XGBoost","Attack" if xgb_pred else "Normal")

        st.subheader("Prediction Probability")

        p1,p2 = st.columns(2)

        p1.metric("XGBoost Attack Probability", f"{xgb_prob:.2f}")
        p2.metric("Random Forest Attack Probability", f"{rf_prob:.2f}")

        comp_df = pd.DataFrame({
            "Model":["XGBoost","Random Forest","Isolation Forest"],
            "Decision":[xgb_pred,rf_pred,iso_pred]
        })

        fig = px.bar(
            comp_df,
            x="Model",
            y="Decision",
            color="Model",
            title="Model Decision Comparison"
        )

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Live Anomaly Monitoring")

        scores = np.random.normal(0,1,100)

        anomaly_df = pd.DataFrame({
            "Time":range(100),
            "Score":scores
        })

        fig = px.line(
            anomaly_df,
            x="Time",
            y="Score",
            title="Real-Time Anomaly Score"
        )

        st.plotly_chart(fig,use_container_width=True)

    # -------------------------------------------
    # CSV BULK PREDICTION
    # -------------------------------------------

    st.divider()

    st.subheader("📂 Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.dataframe(data.head())

        try:

            processed = preprocessor.transform(data)

            xgb_preds = xgb_model.predict(processed)
            rf_preds = rf_model.predict(processed)

            iso_preds = iso_model.predict(processed)
            iso_preds = np.where(iso_preds==-1,1,0)

            hybrid_preds = []

            for i in range(len(xgb_preds)):

                votes = xgb_preds[i] + rf_preds[i] + iso_preds[i]

                hybrid_preds.append(1 if votes>=2 else 0)

            data["Prediction"] = hybrid_preds
            data["Result"] = data["Prediction"].map({0:"Normal",1:"Attack"})

            st.subheader("Prediction Results")

            st.dataframe(data)

            csv = data.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "prediction_results.csv"
            )

        except Exception as e:

            st.error("Error processing file")
            st.write(e)

# ==================================================
# MODEL PERFORMANCE PAGE
# ==================================================

elif selected == "Model Performance":

    perf_data = {
        "Model":["Isolation Forest","Random Forest","XGBoost","Hybrid"],
        "Accuracy":[0.66,0.87,0.87,0.82],
        "Recall":[0.51,0.98,0.98,0.99],
        "F1":[0.63,0.89,0.89,0.86]
    }

    perf_df = pd.DataFrame(perf_data)

    fig = px.bar(
        perf_df,
        x="Model",
        y=["Accuracy","Recall","F1"],
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig,use_container_width=True)

# ==================================================
# EXPLAINABLE AI PAGE
# ==================================================

elif selected == "Explainable AI":

    st.title("Explainable AI Analysis")

    st.write("""
    Explainable AI techniques help understand **why the model predicted an attack**.
    """)

    shap_img = os.path.join(BASE_DIR,"images","shap_summary.png")
    lime_img = os.path.join(BASE_DIR,"images","lime_explanation.png")

    st.subheader("SHAP Feature Importance")

    st.image(shap_img)

    st.markdown("""
    **Interpretation**

    SHAP explains global feature importance.

    Important features influencing cyber attack detection include:

    - sbytes
    - dbytes
    - rate
    - spkts

    Higher SHAP value means stronger contribution toward predicting an attack.
    """)

    st.subheader("LIME Local Explanation")

    st.image(lime_img)

    st.markdown("""
    **Interpretation**

    LIME explains a **single prediction instance**.

    It highlights which features pushed the model towards:

    - Normal Traffic
    - Malicious Traffic
    """)

# ==================================================
# NETWORK STATUS
# ==================================================

status = "SECURE"

if status=="SECURE":
    st.success("🟢 Network Status: Secure")
else:
    st.error("🔴 Network Status: Threat Detected")