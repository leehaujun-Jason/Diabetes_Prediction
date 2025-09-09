import streamlit as st
import numpy as np
import joblib

# Load models
log_reg = joblib.load("logistic_model.pkl")
knn = joblib.load("knn_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes using **Logistic Regression** and **KNN** models.")

# User input form
st.header("Enter Patient Details:")

pregnancies = st.number_input("Number of Pregnancies:", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level (Plasma glucose concentration a 2 hours in an oral glucose tolerance test):", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure (Diastolic blood pressure (mm Hg)):", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness(Triceps skin fold thickness (mm)):", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (2-Hour serum insulin (mu U/ml)):", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (Body mass index (weight in kg/(height in m)^2)):", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age (years):", min_value=1, max_value=100, value=30)

# Prepare input
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Prediction button
if st.button("Predict"):
    # Logistic Regression prediction
    pred_lr = log_reg.predict(input_data)[0]
    result_lr = "Diabetic" if pred_lr == 1 else "Not Diabetic"
    
    # KNN prediction
    pred_knn = knn.predict(input_data)[0]
    result_knn = "Diabetic" if pred_knn == 1 else "Not Diabetic"
    
    st.subheader("Results")
    st.write(f"**Logistic Regression Prediction:** {result_lr}")
    st.write(f"**KNN Prediction:** {result_knn}")
    
    st.info("Note: Predictions are based on trained models and should not replace professional medical advice.")


