import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("best_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# App title
st.title("Employee Attrition Prediction")

# User input
st.header("Enter Employee Data")

# Sample form - adjust fields to match your dataset
department = st.selectbox("Department", ["Sales", "Technical", "HR", "Finance"])
region = st.selectbox("Region", ["region_1", "region_2", "region_3"])  # Example values
education = st.selectbox("Education Level", ["Below Secondary", "Bachelor's", "Master's & above"])
previous_year_rating = st.selectbox("Previous Year Rating", [1, 2, 3, 4, 5])
no_of_trainings = st.slider("Number of Trainings", 1, 10, 1)
age = st.slider("Age", 18, 60, 30)

# Create input DataFrame
input_data = pd.DataFrame({
    "department": [department],
    "region": [region],
    "education": [education],
    "previous_year_rating": [previous_year_rating],
    "no_of_trainings": [no_of_trainings],
    "age": [age],
    # Add other features used in your training data
})

# Predict
if st.button("Predict Attrition"):
    prob = model.predict_proba(input_data)[0][1]  # Get probability of class 1
    threshold = 0.3  # Custom threshold
    prediction = 1 if prob >= threshold else 0

    st.subheader("Prediction:")
    st.write("Attrition" if prediction == 1 else "No Attrition")
    st.write(f"Probability of Attrition: {prob:.2f}")
