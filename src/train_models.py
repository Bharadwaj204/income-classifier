from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from tqdm import tqdm  # For showing the progress bar

def train_all_models(X_train, y_train):
    print("Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Using tqdm to show progress
    for name, model in tqdm(models.items(), desc="Training models", unit="model"):
        model.fit(X_train, y_train)
    
    return models
