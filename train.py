import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load dataset
data = pd.read_csv("data/adult_income.csv")

# Rename columns to match what Streamlit app expects
data.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education Num', 'Marital Status',
                'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
                'Capital Loss', 'Hours/Week', 'Country', 'Target']

# Drop unnecessary column
data.drop(columns=['fnlwgt'], inplace=True)

# Drop rows with missing values (if any)
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Encode categorical columns
categorical_columns = ['Workclass', 'Education', 'Marital Status', 'Occupation',
                       'Relationship', 'Race', 'Sex', 'Country']

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = data[col].astype(str)
    le.fit(data[col])  # Fit on available data (ensure all values are there!)
    data[col] = le.transform(data[col])
    encoders[col] = le

# Encode target
data['Target'] = data['Target'].apply(lambda x: 1 if '>50K' in x else 0)

# Split data
X = data.drop("Target", axis=1)
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (you can try others too)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.4f}")

# Save model and encoders
os.makedirs("models/encoders", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")

for col, encoder in encoders.items():
    joblib.dump(encoder, f"models/encoders/{col}_encoder.pkl")
