import streamlit as st
import numpy as np
import joblib

# Load models and scaler
knn = joblib.load("knn_model.pkl")
#ann = joblib.load("ann_model.pkl")
from tensorflow.keras.models import load_model
ann = load_model("ann_model_keras.keras")

scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes using **KNN** and **ANN** models.")

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

# Scale input with the SAME scaler used during training
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Predict"):
    
    # === KNN prediction ===
    pred_knn = knn.predict(input_scaled)[0]
    pred_proba_knn = knn.predict_proba(input_scaled)[0][1]  # probability of being diabetic
    result_knn_str = "Diabetic" if pred_knn == 1 else "Not Diabetic"

    # === ANN prediction ===
    pred_proba_ann = ann.predict(input_scaled)[0][0]  # ANN usually returns [[prob]]
    pred_ann = int(pred_proba_ann >= 0.5)
    result_ann_str = "Diabetic" if pred_ann == 1 else "Not Diabetic"
    
    # === Display results ===
    st.subheader("Results")
    st.write(f"**KNN Probability of Diabetes:** {pred_proba_knn:.3f}")
    st.write(f"**KNN Prediction:** {result_knn_str}")
    st.write("---")
    st.write(f"**ANN Probability of Diabetes:** {pred_proba_ann:.3f}")
    st.write(f"**ANN Prediction:** {result_ann_str}")
    
    st.info("Note: Predictions are based on trained models and should not replace professional medical advice.")


