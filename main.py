from src.data_preprocessing import load_and_prepare_data
from src.train_models import train_all_models
from src.evaluate_models import evaluate_models
from src.visualization import plot_model_performance

def run_pipeline():
    print("Starting data preprocessing...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("Data preprocessing completed. Starting model training...")
    
    models = train_all_models(X_train, y_train)
    print("Model training completed. Starting model evaluation...")
    
    evaluate_models(models, X_test, y_test)
    print("Model evaluation completed. Generating performance plot...")
    
    plot_model_performance()
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    print("Running the pipeline...")
    run_pipeline()
    print("Process finished.")
