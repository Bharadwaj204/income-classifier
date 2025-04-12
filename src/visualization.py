# src/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_model_performance():
    """Generate a bar plot comparing model performance."""
    df = pd.read_csv("outputs/metrics.csv")
    df.set_index("Model", inplace=True)

    os.makedirs("assets/plots", exist_ok=True)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("assets/plots/model_comparison.png")
