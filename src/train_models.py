# src/train_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from tqdm import tqdm

def train_all_models(X_train, y_train):
    """Train multiple classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for name, model in tqdm(models.items(), desc="Training models", unit="model"):
        model.fit(X_train, y_train)

    return models
