# src/data_preprocessing.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv('data/adult_income.csv')
    df = df.dropna()

    # Drop unnecessary column
    if 'fnlgwt' in df.columns:
        df.drop(columns=['fnlgwt'], inplace=True)

    X = df.drop(columns=['Above/Below 50k'])
    y = df['Above/Below 50k'].apply(lambda x: 1 if x == ' >50K' else 0)

    # Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    encoder_dir = 'models/encoders'
    os.makedirs(encoder_dir, exist_ok=True)
    for col, le in label_encoders.items():
        joblib.dump(le, f'{encoder_dir}/{col}_encoder.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders
