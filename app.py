import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load('models/best_model.pkl')
encoders = {
    col: joblib.load(f'models/encoders/{col}_encoder.pkl')
    for col in ['Country', 'Education', 'Marital Status', 'Occupation', 'Race', 'Relationship', 'Sex', 'Workclass']
}

# Handle encoding with fallback for unseen labels
def safe_label_encode(value, encoder, col_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        st.warning(f"⚠️ '{value}' is unseen in training for {col_name}. Using 'Unknown' fallback.")
        if 'Unknown' not in encoder.classes_:
            # Append 'Unknown' to classes if not present
            encoder.classes_ = np.append(encoder.classes_, 'Unknown')
        return encoder.transform(['Unknown'])[0]

# Preprocessing
def preprocess_input_data(data):
    for col, encoder in encoders.items():
        if col in data.columns:
            data[col] = safe_label_encode(data[col], encoder, col)
        else:
            # Add fallback if column is missing
            st.warning(f"⚠️ Missing column: {col}. Adding default value.")
            data[col] = 'Unknown'  # Fallback for missing column
    return data

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

# Prepare data
input_data = {
    'Age': age,  # Match the column name used during training
    'Workclass': workclass,
    'Education': education,
    'Education Num': education_num,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Relationship': relationship,
    'Race': race,
    'Sex': sex,
    'Capital Gain': capital_gain,  # Match the column name used during training
    'Capital Loss': capital_loss,  # Match the column name used during training
    'Hours/Week': hours_per_week,  # Match the column name used during training
    'Country': native_country  # Match the column name used during training
}

input_df = pd.DataFrame([input_data])

# Rename columns to match those used during training
input_df.rename(columns={
    'Age': 'Age',
    'Workclass': 'Workclass',
    'Education': 'Education',
    'Education Num': 'Education Num',
    'Marital Status': 'Marital Status',
    'Occupation': 'Occupation',
    'Relationship': 'Relationship',
    'Race': 'Race',
    'Sex': 'Sex',
    'Capital Gain': 'Capital Gain',
    'Capital Loss': 'Capital Loss',
    'Hours/Week': 'Hours/Week',
    'Country': 'Country'
}, inplace=True)

# Run prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        processed_input = preprocess_input_data(input_df.copy())
        prediction = model.predict(processed_input)[0]
        result = "Above 50K" if prediction == 1 else "Below 50K"
        st.success(f"💼 Predicted income class: **{result}**")

# Show model metrics
st.subheader("Model Performance (on test data)")
st.write("✅ Gradient Boosting Classifier Accuracy: **0.8709**")
