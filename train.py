import pandas as pd
import joblib
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data_preprocessing import load_and_prepare_data

# Loading the data
print("Loading dataset...")
X_train, X_test, y_train, y_test, label_encoders = load_and_prepare_data()
print("Dataset loaded and preprocessed.")

# Checking the shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(solver='saga', max_iter=2000, random_state=42)
    )
}

# Train, evaluate, and save
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train.values.ravel())  # Avoid shape warning
    print(f"{model_name} trained.")

    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} accuracy: {accuracy:.4f}")

    print(f"Saving the {model_name} model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}_model.pkl')

# Save encoders
print("\nSaving encoders...")
os.makedirs('models/encoders', exist_ok=True)
for col, le in label_encoders.items():
    safe_col = re.sub(r'[^\w\-]', '_', col)
    joblib.dump(le, f'models/encoders/{safe_col}_encoder.pkl')

print("Training and saving complete.")
