import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import requests
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set Streamlit theme
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Add logo and concept image
st.sidebar.image("logo.png", use_column_width=True)
st.title("Heart Disease Prediction App")
st.image("concept_image.png", caption="Understanding Heart Disease Predictions")  # Adjust the width as needed
st.write("""
This application uses machine learning to predict the likelihood of heart disease based on patient data.
It provides technical explanations, including what each feature means and its contribution to the prediction.
""")

# Load models
logistic_model_path = 'logistic_regression_model.pkl'
random_forest_model_path = 'random_forest_model.pkl'

try:
    with open(logistic_model_path, 'rb') as file:
        logistic_model = pickle.load(file)
except FileNotFoundError:
    logistic_model = None

try:
    with open(random_forest_model_path, 'rb') as file:
        random_forest_model = pickle.load(file)
except FileNotFoundError:
    random_forest_model = None

if not logistic_model and not random_forest_model:
    st.error("Both models are missing, please check the files")
    st.stop()

# Sidebar inputs
st.sidebar.title("Input Patient Data")
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1,
                                   help="A measure of ST depression during exercise. Higher values indicate more severe issues.")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3],
                          help="0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic.")
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, step=1,
                                   help="Maximum heart rate achieved during exercise. Higher values are better.")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1,
                               help="Patient's age in years.")
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3],
                          help="Number of major vessels (0–3) colored by fluoroscopy. Higher values indicate better blood flow.")
thal = st.sidebar.selectbox("Thalassemia (thal)", [1, 3, 6, 7],
                            help="1 = Normal, 3 = Fixed Defect, 6 = Reversible Defect, 7 = Abnormal.")

# Prepare input for prediction
try:
    input_dict = {
        "oldpeak": [oldpeak],
        "cp": [cp],
        "thalach": [thalach],
        "age": [age],
        "ca": [ca],
        "thal": [thal],
    }
    input_df = pd.DataFrame(input_dict)
# We use onehotencoder here to make sure the input data has the same format at the trainging data
    encoder = OneHotEncoder(categories=[[0, 1, 2, 3], [1, 3, 6, 7]], drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(input_df[['cp', 'thal']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['cp', 'thal']))
    input_data = pd.concat([input_df.drop(columns=['cp', 'thal']), encoded_df], axis=1)
except Exception as e:
    st.error(f"Error preparing input data: {e}")
    st.stop()

# Sidebar for model selection
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.radio("Select a model", ["Logistic Regression", "Random Forest"])

# Predict button
if st.sidebar.button("Predict"):
    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = random_forest_model
    else:
        st.error("Invalid model choice")
        st.stop()

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"The model predicts heart disease with a probability of {probability[1]:.2f}.")
        else:
            st.success(f"The model predicts no heart disease with a probability of {probability[0]:.2f}.")

        # Feature Explanations
        st.subheader("Feature Explanations")
        st.write("""
        - **Oldpeak**: Higher values indicate more severe ischemia.
        - **Chest Pain Type (cp)**: Differentiates angina from non-anginal chest pain.
        - **Thalach**: Higher maximum heart rates suggest better cardiovascular health.
        - **Age**: Older age is a risk factor for heart disease.
        - **CA**: The number of vessels visualized during fluoroscopy. Higher values indicate better blood flow.
        - **Thal**: Abnormal results suggest blood flow issues or oxygen transport problems.
        """)

        # LLM Interpretation
        try:
            llm_url = "http://127.0.0.1:11434/api/generate"
            llm_prompt = (
                f"The model predicts {'heart disease' if prediction == 1 else 'no heart disease'} "
                f"with a probability of {probability}. Explain this prediction in technical terms, "
                "including the role of Oldpeak, CP, Thalach, CA, and Thal. Like a doctor to a patient"
            )
            response = requests.post(llm_url, json={"model": "llama3.2", "prompt": llm_prompt}, stream=True)

            if response.status_code == 200:
                interpretation = ""
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            parsed_chunk = json.loads(chunk.decode("utf-8"))
                            interpretation += parsed_chunk.get("response", "")
                        except json.JSONDecodeError as e:
                            st.warning(f"Error parsing chunk: {e}")
                if interpretation.strip():
                    st.subheader("LLM Interpretation")
                    st.write(interpretation)
                else:
                    st.warning("LLM interpretation was empty.")
            else:
                st.warning(f"LLM interpretation could not be retrieved. Error: {response.status_code} - {response.text}")
        except Exception as llm_error:
            st.warning(f"Error connecting to LLM: {llm_error}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Feature Importance for Random Forest
if model_choice == "Random Forest" and hasattr(random_forest_model, "feature_importances_"):
    st.subheader("Feature Importance")
    feature_importances = random_forest_model.feature_importances_
    features = [
        'oldpeak', 
        'cp_1', 'cp_2', 'cp_3',
        'thalach', 
        'age', 
        'ca', 
        'thal_3', 'thal_6', 'thal_7'
    ]
    if len(features) != len(feature_importances):
        st.error("Feature list length does not match feature importances length.")
    else:
        plt.figure(figsize=(8, 4))
        plt.barh(features, feature_importances, color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance")
        st.pyplot(plt)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with 45 cups of coffee")
