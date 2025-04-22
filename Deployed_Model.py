import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model pipelines
try:
    detection_model = joblib.load("land_mines_randomforest_detection.joblib")
    classification_model = joblib.load("land_mines_randomforest_classification.joblib")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("Land Mine Detection App (Hopefully ;)")
st.markdown(
    "This app predicts whether a land mine is present or not based on various sensory inputs; "
    "well I mean hopefully it does... It's not like you are gonna come complaining... right?"
)

soil_type_values = {
    1: "Dry and Sandy",
    2: "Dry and Humus",
    3: "Dry and Limy",
    4: "Humid and Sandy",
    5: "Humid and Humus", 
    6: "Humid and Limy"
}

v = st.number_input(
    "Output voltage of FLC sensor due to magnetic distortion:",
    min_value=0.0, max_value=10.6, step=0.1, value=5.0
)

h = st.number_input(
    "Enter the height of the sensor from the ground (cm):",
    min_value=0.0, max_value=20.0, step=0.5, value=10.0
)

s = st.selectbox(
    "Soil type depending on moisture content:",
    options=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    format_func=lambda x: f"Type {x} ({x*5:.0f}/5 humidity)"
)

if st.button("To mine or not to mine?"):
    try:
        # Create input dataframe matching the original dataset format
        input_data = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
        
        # Debug information
        st.info(f"Input data: V={v}, H={h}, S={s}")
        
        # First, predict if it's a mine or not (binary classification)
        is_mine = None
        try:
            # Use the detection model for binary classification
            is_mine = detection_model.predict(input_data)[0]
            st.write(f"Is mine detection result: {is_mine}")
            
        except Exception as e:
            st.error(f"Error in mine detection: {e}")
            
            # Try manual preprocessing as fallback
            try:
                st.warning("Trying manual preprocessing...")
                # Create the feature matrix manually
                X = pd.DataFrame({
                    "V": [v],
                    "H": [h]
                })
                
                # One-hot encode S manually
                for cat in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    X[f"S_{cat}"] = 1 if s == cat else 0
                    
                # Try prediction with manual features
                is_mine = detection_model.named_steps['classifier'].predict(X)[0]
                st.write(f"Fallback detection result: {is_mine}")
            except Exception as e2:
                st.error(f"Fallback detection also failed: {e2}")
        
        # Based on the detection result, classify the mine type
        if is_mine == 1:  # If a mine is detected
            try:
                # Use the classification model to predict the type
                mine_class = classification_model.predict(input_data)[0]
                
                mine_classes = {
                    1: "No Mine (Class 1)",
                    2: "AT Mine (Class 2)",
                    3: "AP Mine (Class 3)",
                    4: "Booby-trapped AP (Class 4)",
                    5: "M14 AP (Class 5)"
                }
                
                st.success(f"Prediction: {mine_classes.get(mine_class, 'Unknown')}")
                
            except Exception as e:
                st.error(f"Error in mine classification: {e}")
                st.warning("Unable to classify mine type. Please check model compatibility.")
        else:
            st.success("Prediction: No mine detected (Class 1)")
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("Debug tip: Check sklearn version compatibility between training and deployment.")
