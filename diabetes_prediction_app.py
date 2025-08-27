{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8559d037-2807-492e-94ac-9b15263a1388",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load models\n",
    "log_reg = joblib.load(\"models/logistic_model.pkl\")\n",
    "knn = joblib.load(\"models/knn_model.pkl\")\n",
    "\n",
    "st.title(\"🩺 Diabetes Prediction App\")\n",
    "st.write(\"This app predicts the likelihood of diabetes using **Logistic Regression** and **KNN** models.\")\n",
    "\n",
    "# User input form\n",
    "st.header(\"Enter Patient Details:\")\n",
    "\n",
    "pregnancies = st.number_input(\"Number of Pregnancies:\", min_value=0, max_value=20, value=1)\n",
    "glucose = st.number_input(\"Glucose Level:\", min_value=0, max_value=300, value=120)\n",
    "blood_pressure = st.number_input(\"Blood Pressure:\", min_value=0, max_value=200, value=70)\n",
    "skin_thickness = st.number_input(\"Skin Thickness:\", min_value=0, max_value=100, value=20)\n",
    "insulin = st.number_input(\"Insulin Level:\", min_value=0, max_value=900, value=80)\n",
    "bmi = st.number_input(\"BMI:\", min_value=0.0, max_value=70.0, value=25.0)\n",
    "dpf = st.number_input(\"Diabetes Pedigree Function:\", min_value=0.0, max_value=3.0, value=0.5)\n",
    "age = st.number_input(\"Age:\", min_value=1, max_value=120, value=30)\n",
    "\n",
    "# Prepare input\n",
    "input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])\n",
    "\n",
    "# Prediction button\n",
    "if st.button(\"Predict\"):\n",
    "    # Logistic Regression prediction\n",
    "    pred_lr = log_reg.predict(input_data)[0]\n",
    "    result_lr = \"Diabetic\" if pred_lr == 1 else \"Not Diabetic\"\n",
    "    \n",
    "    # KNN prediction\n",
    "    pred_knn = knn.predict(input_data)[0]\n",
    "    result_knn = \"Diabetic\" if pred_knn == 1 else \"Not Diabetic\"\n",
    "    \n",
    "    st.subheader(\"Results\")\n",
    "    st.write(f\"**Logistic Regression Prediction:** {result_lr}\")\n",
    "    st.write(f\"**KNN Prediction:** {result_knn}\")\n",
    "    \n",
    "    st.info(\"Note: Predictions are based on trained models and should not replace professional medical advice.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
