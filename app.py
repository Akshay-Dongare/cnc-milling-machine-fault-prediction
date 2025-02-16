import streamlit as st
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('MLGuard.keras')

# Load the LabelEncoder
label_encoder = joblib.load('label_encoder.joblib')

# Load the pre-fitted scaler
scaler = joblib.load('scaler.joblib')

# Define the feature names
feature_names = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Define the failure types
failure_types = label_encoder.classes_

# Streamlit app
st.set_page_config(page_title="MLGuard Failure Prediction", layout="centered")
st.title("MLGuard: Failure Type Prediction")

# Input fields for each feature
st.header("Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=10000.0, value=0.0, step=100.0)

# Convert input data to a DataFrame with the same column names
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Standardize the input data using the pre-fitted scaler
input_data_scaled = scaler.transform(input_data_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_probabilities = prediction[0]

    # Display the result
    if predicted_class == 0:  # Assuming 'No_Failure' is class 0
        st.write(f"Prediction: {failure_types[predicted_class]}")
    else:
        st.write(f"Prediction: {failure_types[predicted_class]}")

    # Display probabilities
    st.subheader("Failure Type Probabilities:")
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    for idx in sorted_indices:
        st.write(f"{failure_types[idx]}: {predicted_probabilities[idx] * 100:.2f}%")

    # Display a bar chart for probabilities
    probabilities_df = pd.DataFrame(predicted_probabilities * 100, index=failure_types, columns=["Probability"])
    st.bar_chart(probabilities_df)

# Footer
st.markdown("---")
st.markdown("Developed by Team Magenta | Powered by Streamlit and TensorFlow")