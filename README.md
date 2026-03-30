XAI-Based Hybrid Zero-Day Intrusion Detection System

Overview

This project presents a hybrid machine learning-based Intrusion Detection System (IDS) designed to detect zero-day cyber attacks. The system integrates both unsupervised anomaly detection and supervised classification models to enhance detection performance.

Additionally, Explainable Artificial Intelligence (XAI) techniques are incorporated to provide transparency and interpretability of predictions, making the system more reliable for real-world cybersecurity applications.

---

Objectives

- Detect unknown (zero-day) cyber attacks effectively
- Improve detection accuracy using hybrid modeling
- Provide explainable insights for decision-making
- Enable real-time monitoring through an interactive dashboard

---

Methodology

Data Preprocessing

- Handling missing values
- Encoding categorical variables
- Feature scaling and normalization
- Feature selection for optimal performance

Machine Learning Models

The system uses a hybrid combination of:

- Isolation Forest → Detects anomalies (zero-day attacks)
- Random Forest → Classifies network traffic
- XGBoost → Enhances prediction accuracy using boosting

Hybrid Decision Engine

- Combines outputs from multiple models
- Improves robustness in detecting both known and unknown attacks

---

Explainable AI (XAI)

To improve transparency and interpretability:

- SHAP (SHapley Additive Explanations)
  
  - Provides global feature importance
  - Helps understand overall model behavior

- LIME (Local Interpretable Model-agnostic Explanations)
  
  - Explains individual predictions
  - Shows why a sample is classified as attack or normal

---

Performance Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC

---

Dataset

- UNSW-NB15 Dataset
- Contains real-world network traffic data

Note: The dataset is not included in this repository due to size limitations.

Download the dataset from the official source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Instructions to Access the Dataset

1. Open the above link in a browser
2. Scroll down to find the sentence containing the dataset access link
3. Click on the "HERE" hyperlink provided on the page
4. This will redirect you to the dataset repository
5. Navigate to the CSV Files section
6. Open the folder named "Training and Testing Sets"
7. Download the following files:
   - UNSW_NB15_training-set.csv
   - UNSW_NB15_testing-set.csv

After downloading, place the files inside a "data/" directory in your project.

---

Project Structure

XAI-HYBRID-ZERODAY-IDS_SYSTEM/
│
├── src/                 # Core ML models and preprocessing
├── notebooks/           # Jupyter notebooks for analysis
├── ui/                  # Streamlit dashboard
├── train_models.py      # Model training script
├── README.md            # Project documentation

---

How to Run

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Train the models

python train_models.py

Step 3: Run the dashboard

streamlit run ui/app.py

---

Visualization and Dashboard

- Real-time intrusion detection
- Feature importance visualization
- Model predictions display

---

Key Features

- Hybrid ML approach (Anomaly + Classification)
- Zero-day attack detection
- Explainable AI (SHAP + LIME)
- Interactive Streamlit dashboard
- Improved accuracy and robustness

---

Limitations

- Dataset not included due to size constraints
- Performance depends on quality of training data
- Requires computational resources for training

---

Future Work

- Integration of deep learning models
- Real-time network data streaming
- Cloud-based deployment
- Advanced explainability techniques

---

Author

SalvaSoumya

---

Conclusion

This project demonstrates a robust hybrid IDS capable of detecting zero-day attacks with improved accuracy and interpretability. The integration of Explainable AI enhances transparency, making the system more suitable for practical cybersecurity applications.

---
