import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    det_pipeline = joblib.load("land_mines_randomforest_detection.joblib")
    clf_pipeline = joblib.load("land_mines_randomforest_classification.joblib")
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

s = int(st.number_input(
    "Soil type depending on moisture content (1-6):",
    min_value=1, max_value=6, step=1, value=3
))

if st.button("To mine or not to mine?"):
    try:
        input_data = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
        
        scaled_data = pd.DataFrame(input_data.copy())
        
        for cat in range(2, 7): 
            col_name = f"S_{cat}"
            scaled_data[col_name] = 1 if s == cat else 0
            
        if "S" in scaled_data.columns:
            scaled_data = scaled_data.drop("S", axis=1)
            
        try:
            preprocessor = det_pipeline.named_steps['preprocess']
            scaler = preprocessor.transformers_[0][1]
            scaled_vh = scaler.transform(input_data[["V", "H"]])
            scaled_data["V"] = scaled_vh[0][0]
            scaled_data["H"] = scaled_vh[0][1]
            
            is_mine_detected = det_pipeline.named_steps['classifier'].predict(scaled_data)[0]
            
            if is_mine_detected == 1:  
                mine_class = clf_pipeline.named_steps['classifier'].predict(scaled_data)[0]
                
                mine_classes = {
                    1: "No Mine (Class 1)",
                    2: "AT Mine (Class 2)",
                    3: "AP Mine (Class 3)",
                    4: "Booby-trapped AP (Class 4)",
                    5: "M14 AP (Class 5)"
                }
                st.success(f"Prediction: {mine_classes.get(mine_class, 'Unknown')}")
            else:
                st.success("Prediction: No mine detected (Class 1)")
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            
            raw = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
            for cat in range(2, 7):
                raw[f"S_{cat}"] = (raw["S"] == cat).astype(int)
            X = raw.drop("S", axis=1)
            
            try:
                st.warning("Using fallback prediction method with unscaled data.")
                is_mine = det_pipeline.named_steps['classifier'].predict(X)[0]
                
                mine_classes = {
                    1: "No Mine (Class 1)",
                    2: "AT Mine (Class 2)",
                    3: "AP Mine (Class 3)",
                    4: "Booby-trapped AP (Class 4)",
                    5: "M14 AP (Class 5)"
                }
                st.success(f"Prediction: {mine_classes.get(is_mine, 'Unknown')}")
            except Exception as e2:
                st.error(f"Fallback prediction also failed: {e2}")
                st.info("Please check your model pipeline structure and try again.")
                
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Debug info: Make sure you're using the same version of scikit-learn for both training and deployment.")
