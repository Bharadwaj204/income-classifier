import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_model_performance():
    print("Generating performance plot...")
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
    print("Plot saved to assets/plots/model_comparison.png")
