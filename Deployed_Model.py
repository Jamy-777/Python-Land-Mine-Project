import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
detection_model = joblib.load("land_mines_randomforest_detection.joblib")

# Extract the classifier from the pipeline
# The pipeline typically has two steps: preprocessor and classifier
preprocessor = detection_model.named_steps['preprocess']
classifier = detection_model.named_steps['classifier']

# Custom prediction function that handles preprocessing
def safe_predict(input_data):
    try:
        # Extract only V and H features which are numeric and not causing issues
        numeric_features = input_data[['V', 'H']].copy()
        
        # Create dummy variables for S manually
        # Assuming S was originally encoded as 1.0, 2.0, 3.0, 4.0
        s_value = input_data['S'].iloc[0]
        
        # Create one-hot encoding manually
        s_one_hot = np.zeros(3)  # 4 categories - 1 (drop_first=True)
        if s_value == 2.0:
            s_one_hot[0] = 1
        elif s_value == 3.0:
            s_one_hot[1] = 1
        elif s_value == 4.0:
            s_one_hot[2] = 1
        # If s_value is 1.0, all values remain 0 due to drop_first=True
        
        # Scale the numeric features
        # Get the StandardScaler for V and H
        scaler = preprocessor.named_transformers_['scale']
        scaled_numeric = scaler.transform(numeric_features)
        
        # Combine scaled numeric with one-hot encoded features
        final_features = np.hstack([scaled_numeric, s_one_hot.reshape(1, -1)])
        
        # Predict using the classifier directly
        prediction = classifier.predict(final_features)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

st.title("Land Mine Detection App (Hopefully ;)")
st.markdown("This app predicts whether a land mine is present or not based on various sensory inputs, " \
"well I mean hopefully it does... It's not like you are gonna come complaining... right?")

soil_type_values = {
    1: "Dry and Sandy",
    2: "Dry and Humus",
    3: "Dry and Limy",
    4: "Humid and Sandy"
}

v = st.number_input("Output voltage of FLC sensor due to magnetic distortion:", 
                    min_value=0.0, max_value=10.6, step=0.1, value=5.0)

h = st.number_input("Enter the height of the sensor from the ground(cm):", min_value=0.0, max_value=20.0, 
                    step=0.5, value=10.0)

s = st.number_input("Soil type depending on moisture content(1-4):", min_value=1.0, max_value=4.0, step=1.0,
                    value=1.0)

if st.button("To mine or not to mine?"):
    input_data = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
    
    # Use our custom predict function
    is_mine = safe_predict(input_data)
    
    if is_mine == 1:
        st.success("Mine detected!")
    else:
        st.success("No mine detected (Safe)")
