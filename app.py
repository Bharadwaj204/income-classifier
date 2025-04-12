import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Base path for consistent file access
base_path = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(base_path, 'models', 'best_model.pkl')
model = joblib.load(model_path)

# Load encoders
encoder_columns = ['Country', 'Education', 'Marital Status', 'Occupation', 'Race', 'Relationship', 'Sex', 'Workclass']
encoders = {
    col: joblib.load(os.path.join(base_path, 'models', 'encoders', f'{col}_encoder.pkl'))
    for col in encoder_columns
}

# Handle encoding with fallback for unseen labels
def safe_label_encode(value, encoder, col_name):
    # Ensure value is passed as a list for consistent processing
    value_list = [value] if not isinstance(value, list) else value
    
    if value_list[0] in encoder.classes_:
        return encoder.transform(value_list)[0]
    else:
        st.warning(f"⚠️ '{value_list[0]}' is unseen for {col_name}. Using fallback index 0.")
        return 0  # Fallback to index 0 (e.g., 'Private', 'White', etc.)

# Preprocessing function
def preprocess_input_data(data):
    for col, encoder in encoders.items():
        if col in data.columns:
            # Make sure to apply the encoding safely
            data[col] = safe_label_encode(data[col].iloc[0], encoder, col)
        else:
            st.warning(f"⚠️ Missing column: {col}. Adding default value.")
            data[col] = 0  # Fallback if column is missing
    return data

# Ensure all features from the model are in the input data
def ensure_all_features(input_data, model):
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_data.columns:
            st.warning(f"⚠️ Missing feature '{feature}'. Adding default value.")
            input_data[feature] = 0  # Fallback for missing features
    return input_data[model_features]  # Ensure feature order matches training

# Streamlit UI
st.set_page_config(page_title="Income Classifier", layout="centered")
st.title("Income Classification Predictor")
st.write("Enter the details below to predict if income is above or below $50K.")

# Input fields
age = st.number_input("Age", min_value=17, max_value=100, value=30)
workclass = st.selectbox("Workclass", encoders['Workclass'].classes_)
education = st.selectbox("Education", encoders['Education'].classes_)
education_num = st.number_input("Education Num", min_value=1, max_value=16, value=9)
marital_status = st.selectbox("Marital Status", encoders['Marital Status'].classes_)
occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['Relationship'].classes_)
race = st.selectbox("Race", encoders['Race'].classes_)
sex = st.selectbox("Sex", encoders['Sex'].classes_)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country", encoders['Country'].classes_)

# Prepare input DataFrame
input_data = {
    'Age': age,
    'Workclass': workclass,
    'Education': education,
    'Education Num': education_num,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Relationship': relationship,
    'Race': race,
    'Sex': sex,
    'Capital Gain': capital_gain,
    'Capital Loss': capital_loss,
    'Hours/Week': hours_per_week,
    'Country': native_country
}
input_df = pd.DataFrame([input_data])

# Prediction logic
if st.button("Predict"):
    with st.spinner("Predicting..."):
        processed_input = preprocess_input_data(input_df.copy())

        # Ensure all features are present in the input data
        processed_input = ensure_all_features(processed_input, model)

        # Prediction
        prediction = model.predict(processed_input)[0]
        result = "Above 50K" if prediction == 1 else "Below 50K"
        st.success(f"💼 Predicted income class: **{result}**")

# Optional model performance display
st.subheader("Model Performance (on test data)")
st.write("✅ Gradient Boosting Classifier Accuracy: **0.8709**")
