# src/evaluate_models.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test data and save the results."""
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    df_metrics = pd.DataFrame(metrics)
    os.makedirs("outputs", exist_ok=True)
    df_metrics.to_csv("outputs/metrics.csv", index=False)
