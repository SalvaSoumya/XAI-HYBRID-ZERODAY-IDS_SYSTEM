import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
def handle_missing_values(df):
    df = df.copy()
    df.fillna(0, inplace=True)
    return df
def encode_categorical(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns
    
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])
    
    return df
def zero_day_split(df, train_attacks, test_attacks):
    train_df = df[df['attack_cat'].isin(train_attacks)]
    test_df = df[df['attack_cat'].isin(test_attacks)]
    return train_df, test_df
def split_features_labels(df):
    X = df.drop(['label', 'attack_cat'], axis=1)
    y = df['label']
    return X, y
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
