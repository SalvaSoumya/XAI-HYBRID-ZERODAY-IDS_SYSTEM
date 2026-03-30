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

## Dataset

This project uses the UNSW-NB15 dataset, a comprehensive and modern network intrusion detection dataset developed by the University of New South Wales (UNSW Canberra). The dataset contains realistic network traffic with both normal and malicious activities, making it suitable for evaluating intrusion detection systems, especially for detecting zero-day attacks.

The dataset includes a wide range of attack categories such as DoS, Exploits, Reconnaissance, Shellcode, and Generic attacks, along with normal traffic. It provides rich feature representations extracted using the IXIA PerfectStorm tool, enabling both anomaly detection and classification-based approaches.

Due to its large size, the dataset is not included directly in this repository.

---

### Dataset Access

You can access the dataset using the following links:

Official UNSW Dataset Page:  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Google Drive link for the dataset:  
https://drive.google.com/drive/folders/1e_HZF2vxGjYwr4ix3-3cPx93rmfY0AzM?usp=drive_link

---

### Instructions to Download

1. Open the official UNSW dataset link  
2. Click on the download section or "Here" link provided on the page  
3. Navigate to the CSV Files directory  
4. Open the "Training and Testing Sets" folder  
5. Download the following files:
   - UNSW_NB15_training-set.csv  
   - UNSW_NB15_testing-set.csv  

Alternatively, you can directly download the dataset from the provided Google Drive link.

---

### Dataset Placement

After downloading, place the dataset files in the following directory:

data/raw/

Required files:

- UNSW_NB15_training-set.csv  
- UNSW_NB15_testing-set.csv  

---

### Dataset Description

- Total Records: Approximately 2.5 million network flows  
- Features: 49 features including flow-based, content-based, time-based, and additional generated features  
- Classes: Binary classification (Normal / Attack) and multi-class attack categories  
- Data Type: Tabular (CSV format)  

---

### Notes

- The dataset is hosted externally due to GitHub file size limitations  
- Ensure correct file placement before running training scripts  
- Preprocessing steps such as encoding and scaling are handled within the project pipeline
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
