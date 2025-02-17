import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# # Load the trained model
# model = tf.keras.models.load_model('MLGuard.keras')

# load trained models
tree_model = joblib.load('best_tree.joblib')
mlp_model = joblib.load('mlp_model.joblib')

# Load the LabelEncoder (only relevant for model 2)
label_encoder = joblib.load('new_label_encoder.joblib')

# Load the pre-fitted scalers
scaler1 = joblib.load('model1_scaler.joblib')
scaler2 = joblib.load('model2_scaler.joblib')

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
targets = {0: 'No_Failure', 1: 'Failure'}

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
input_data_scaled = scaler1.transform(input_data_df)
input_data_scaled_2 = scaler2.transform(input_data_df)

# Make prediction for first model
if st.button("Predict"):
    prediction = tree_model.predict(input_data_scaled)
    predicted_class = prediction[0]

    # display result
    st.write(f"Prediction: {targets[predicted_class]}")

    if predicted_class == 1:

        # run neural net
        failure_prediction = mlp_model.predict(input_data_scaled_2)[0]
        failure_probas = mlp_model.predict_proba(input_data_scaled_2)[0]

        # Display the result
        st.write(f"Predicted Failure Type: {failure_types[failure_prediction]}")

        # Display probabilities
        st.subheader("Failure Type Probabilities:")
        sorted_indices = np.argsort(failure_probas)[::-1]
        for idx in sorted_indices:
            st.write(f"{failure_types[idx]}: {failure_probas[idx] * 100:.2f}%")
        # for id, prob in enumerate(failure_probas):
        #     st.write(f'{label_encoder.inverse_transform([id])[0]}: {prob:.6f}')

        # Display a bar chart for probabilities
        # probabilities_df = pd.DataFrame(failure_probas * 100, index=failure_types, columns=["Probability"])
        probabilities_data = [{'Failure Type': failure_types[i], 'Probability': prob * 100} for i, prob in enumerate(failure_probas)]
        probabilities_df = pd.DataFrame(probabilities_data)
        probabilities_df = probabilities_df.set_index('Failure Type')
        st.bar_chart(probabilities_df)

# Footer
st.markdown("---")
st.markdown("Developed by Team Magenta | Powered by Streamlit and Scikit")