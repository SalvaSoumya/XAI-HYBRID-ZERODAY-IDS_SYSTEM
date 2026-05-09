# Explainable AI-Based Hybrid Zero-Day Intrusion Detection System

An advanced hybrid Intrusion Detection System (IDS) designed to detect both known and unknown cyber attacks using Machine Learning, Anomaly Detection, and Explainable AI techniques.

The system combines supervised and unsupervised learning models to improve intrusion detection accuracy while providing transparent and interpretable predictions through SHAP and LIME.

---

## Overview

Traditional Intrusion Detection Systems struggle to detect zero-day attacks because they rely heavily on predefined attack signatures.

This project introduces a hybrid IDS framework that combines:

- Isolation Forest for anomaly detection
- Random Forest for supervised classification
- XGBoost for boosted classification performance
- Explainable AI techniques for model transparency

The system is capable of detecting both known and previously unseen cyber attacks while providing interpretable insights into model decisions.

---

## Objectives

- Detect zero-day cyber attacks effectively
- Improve intrusion detection accuracy
- Reduce false negatives and false positives
- Provide explainable AI-based insights
- Support real-time intrusion monitoring
- Build an interactive cybersecurity dashboard

---

## Key Features

- Hybrid Machine Learning approach
- Zero-day attack detection
- Real-time intrusion analysis
- Explainable AI integration
- SHAP-based feature importance
- LIME-based prediction explanation
- Interactive Streamlit dashboard
- Network traffic visualization
- High recall for attack detection

---

## Machine Learning Models

### Isolation Forest
- Unsupervised anomaly detection
- Detects unknown and zero-day attacks

### Random Forest
- Supervised classification model
- Detects known attack patterns

### XGBoost
- Gradient boosting classification
- Improves prediction performance and robustness

---

## Hybrid Detection Mechanism

The system combines predictions from:

- Isolation Forest
- Random Forest
- XGBoost

A voting-based hybrid decision engine is used to generate the final prediction, improving overall detection capability and reducing classification errors.

---

## Explainable AI (XAI)

The project integrates Explainable Artificial Intelligence techniques to improve transparency and trust in cybersecurity predictions.

### SHAP (SHapley Additive Explanations)
- Global feature importance analysis
- Explains overall model behavior
- Identifies most influential network traffic features

### LIME (Local Interpretable Model-Agnostic Explanations)
- Explains individual predictions
- Shows why a network sample is classified as attack or normal

---

## Dataset

This project uses the **UNSW-NB15 dataset**, a modern network intrusion detection dataset developed by UNSW Canberra.

The dataset includes:

- Normal network traffic
- Malicious traffic
- Multiple attack categories:
  - DoS
  - Exploits
  - Reconnaissance
  - Shellcode
  - Generic attacks

### Dataset Features

- Approximately 2.5 million network flows
- 49 network traffic features
- Binary classification:
  - Normal Traffic
  - Attack Traffic

---

## Dataset Access

### Official UNSW Dataset

https://research.unsw.edu.au/projects/unsw-nb15-dataset

### Google Drive Dataset Link

https://drive.google.com/drive/folders/1e_HZF2vxGjYwr4ix3-3cPx93rmfY0AzM?usp=sharing

---

## Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Feature scaling
- Feature normalization
- Categorical encoding
- Feature selection
- Data cleaning

Selected important features include:

- proto
- service
- state
- spkts
- dpkts
- sbytes
- dbytes
- rate
- sttl
- dttl

---

## Performance Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Model Performance

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|------|------|------|------|------|
| Isolation Forest | 66.83% | 51.66% | 63.17% | 0.7908 |
| Random Forest | 87.04% | 98.65% | 89.34% | 0.9794 |
| XGBoost | 87.04% | 98.65% | 89.34% | 0.9794 |
| Hybrid Model | 82.26% | 99.32% | 86.64% | 0.9831 |

The hybrid model achieved the highest recall, making it highly effective for detecting malicious network traffic with reduced false negatives.

---

## System Architecture

The system workflow includes:

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Hybrid Decision Engine
6. Explainable AI Analysis
7. Real-Time Dashboard Visualization

---

## Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- XGBoost
- Isolation Forest

### Explainable AI
- SHAP
- LIME

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Dashboard
- Streamlit

### Development Tools
- Jupyter Notebook

---

## Project Structure

```bash
XAI-HYBRID-ZERODAY-IDS_SYSTEM/
│
├── data/
│   └── raw/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── xai/
│   └── hybrid_engine/
│
├── notebooks/
│
├── ui/
│   └── app.py
│
├── train_models.py
├── requirements.txt
├── README.md
└── results/
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/SalvaSoumya/XAI-HYBRID-ZERODAY-IDS_SYSTEM.git

cd XAI-HYBRID-ZERODAY-IDS_SYSTEM
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download Dataset

Download:

- UNSW_NB15_training-set.csv
- UNSW_NB15_testing-set.csv

Place files inside:

```bash
data/raw/
```

---

### 4. Train Models

```bash
python train_models.py
```

---

### 5. Run Dashboard

```bash
streamlit run ui/app.py
```

---

## Dashboard Features

- Real-time intrusion prediction
- Feature importance visualization
- Attack detection results
- Interactive cybersecurity analytics
- Model prediction explanations

---

## Results & Analysis

- Random Forest and XGBoost achieved high classification accuracy
- Isolation Forest improved zero-day anomaly detection
- Hybrid model achieved highest recall
- Explainable AI improved model transparency
- Streamlit dashboard enabled interactive monitoring

---

## Limitations

- Dataset is not included due to size constraints
- Requires computational resources for model training
- Performance depends on dataset quality and preprocessing

---

## Future Enhancements

- Deep learning-based intrusion detection
- Real-time network packet streaming
- Cloud deployment
- Advanced explainability techniques
- IoT network intrusion detection support

---

## License

This project is developed for academic and educational purposes.

---

## Author

Salva Soumya

GitHub: https://github.com/SalvaSoumya
