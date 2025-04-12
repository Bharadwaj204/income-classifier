# src/tune_models.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

def tune_models(X_train, y_train):
    """Tune models with GridSearchCV."""
    results = []

    # Logistic Regression
    logreg_params = {'C': [0.01, 0.1, 1, 10]}
    logreg = GridSearchCV(LogisticRegression(max_iter=1000), logreg_params, cv=5)
    logreg.fit(X_train, y_train)
    results.append({
        'Model': 'Logistic Regression',
        'Best Score': logreg.best_score_,
        'Best Params': logreg.best_params_
    })

    # Decision Tree
    dt_params = {'max_depth': [3, 5, 10, None]}
    dt = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
    dt.fit(X_train, y_train)
    results.append({
        'Model': 'Decision Tree',
        'Best Score': dt.best_score_,
        'Best Params': dt.best_params_
    })

    # Random Forest
    rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)
    rf.fit(X_train, y_train)
    results.append({
        'Model': 'Random Forest',
        'Best Score': rf.best_score_,
        'Best Params': rf.best_params_
    })

    # Save results
    df = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/tuning_results.csv", index=False)
