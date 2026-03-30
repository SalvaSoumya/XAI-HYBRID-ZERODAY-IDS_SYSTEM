import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Drop columns that cause leakage
    columns_to_drop = []

    if 'id' in df.columns:
        columns_to_drop.append('id')

    if 'attack_cat' in df.columns:
        columns_to_drop.append('attack_cat')

    df = df.drop(columns=columns_to_drop)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Separate features and label
    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y