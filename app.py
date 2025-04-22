import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# =======================================
# Define Custom Transformers
# =======================================
class EducationImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['education'] = X['education'].fillna('Unknown')
        return X

class PreviousYearRatingImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['previous_year_rating'] = X['previous_year_rating'].fillna(1.0)
        return X

class EducationOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder(categories=[['Below Secondary', "Bachelor's", "Master's & above"]])
    
    def fit(self, X, y=None):
        self.encoder.fit(X[['education']])
        return self
    
    def transform(self, X):
        X['education'] = self.encoder.transform(X[['education']])
        return X

# =======================================
# Load final XGBoost model pipeline
# =======================================
def load_model():
    with open("best_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
best_threshold = 0.6

# =======================================
# Streamlit App Interface
# =======================================
st.title("HR Promotion Prediction")
st.markdown("Enter employee data to predict promotion eligibility.")

# =======================================
# Input Form
# =======================================
with st.form("promotion_form"):
    department = st.selectbox("Department", [
        "Sales & Marketing", "Operations", "Technology", "HR", "Finance", "Procurement", "R&D"
    ])
    region = st.selectbox("Region", [f"region_{i}" for i in range(40)])
    education = st.selectbox("Education", ["Below Secondary", "Bachelor's", "Master's & above"])
    previous_year_rating = st.selectbox("Previous Year Rating", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    no_of_trainings = st.slider("Number of Trainings", 1, 10, 1)
    age = st.slider("Age", 20, 60, 30)
    length_of_service = st.slider("Length of Service (Years)", 1, 35, 5)
    avg_training_score = st.slider("Average Training Score", 40, 100, 70)
    awards_won = st.selectbox("Awards Won?", [0, 1])

    submitted = st.form_submit_button("Predict Promotion")

# =======================================
# Perform Prediction
# =======================================
if submitted:
    input_data = pd.DataFrame([{
        "department": department,
        "region": region,
        "education": education,
        "previous_year_rating": previous_year_rating,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "length_of_service": length_of_service,
        "avg_training_score": avg_training_score,
        "awards_won": awards_won
    }])
    
# =======================================
# Load final XGBoost model pipeline
# =======================================
# Ensure this is ALL indented under the if-block
    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= best_threshold)

    st.subheader("Prediction Result:")
    st.write("✅ Promoted" if prediction == 1 else "❌ Not Promoted")
    st.write(f"Promotion Probability: **{proba:.2%}** (Threshold = {best_threshold})")


