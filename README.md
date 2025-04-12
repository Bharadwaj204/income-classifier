# Income Classifier

This project is a machine learning model to classify income based on various demographic attributes. The goal is to predict whether a person earns above or below 50K based on their features.

## Features
- Various classification models such as Logistic Regression, Random Forest, Support Vector Machine (SVM), etc.
- Data preprocessing, feature engineering, and model evaluation.
- Visualizations comparing the performance of different models.

## Setup

### Prerequisites

Make sure you have Python 3.6 or higher installed. You can download it from the official website: [python.org](https://www.python.org/downloads/).

Additionally, you'll need to install the required dependencies. You can do this by running:

```bash
├── assets/                  # Contains plots and visualizations
│   └── plots/               # Model comparison and other plots
├── data/                    # Dataset used for training the model
│   └── adult_income.csv     # Dataset file
├── models/                  # Trained models and encoders
│   ├── best_model.pkl       # Best performing model
│   └── encoders/            # Encoders for categorical variables
├── outputs/                 # Metrics and model evaluation results
│   └── metrics.csv          # Performance metrics
├── src/                     # Python scripts for preprocessing, model training, evaluation
│   ├── data_preprocessing.py # Data preprocessing script
│   ├── evaluate_models.py   # Model evaluation script
│   ├── train_models.py      # Model training script
│   ├── tune_models.py       # Hyperparameter tuning script
│   └── visualization.py     # Visualization script for model comparison
├── app.py                   # Main application to run the model
├── main.py                  # Entry script for running the project
├── requirements.txt         # Python dependencies
├── train.py                 # Script for training the models
└── README.md                # Project documentation
