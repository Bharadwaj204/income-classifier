import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Function to load and prepare the dataset
def load_and_prepare_data():
    # Load dataset
    df = pd.read_csv('data/adult_income.csv')
    
    # Clean data (Remove missing values)
    df = df.dropna()
    
    # Define the features and target variable
    X = df.drop(columns=['Above/Below 50k'])
    y = df['Above/Below 50k'].apply(lambda x: 1 if x == ' >50K' else 0)
    
    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Save label encoders for later use
    encoder_dir = 'models/encoders'
    for col, le in label_encoders.items():
        joblib.dump(le, f'{encoder_dir}/{col}_encoder.pkl')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders

# Function to preprocess input data for prediction
def preprocess_input_data(data: pd.DataFrame, label_encoders: dict):
    for col, encoder in label_encoders.items():
        if col in data:
            # Handle unseen labels gracefully
            if data[col] in encoder.classes_:
                data[col] = encoder.transform([data[col]])[0]  # Transform the input data
            else:
                # Handle unseen labels by assigning a default value (e.g., "Unknown")
                data[col] = encoder.transform(['Unknown'])[0]  # Use a label that exists in the encoder
    return data
