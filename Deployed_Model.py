import streamlit as st
import pandas as pd
import joblib

# Load both models
detection_model = joblib.load("land_mines_randomforest_detection.joblib")
classification_model = joblib.load("land_mines_randomforest_classification.joblib")  # Load the classification model too

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
                    value=3.0)

if st.button("To mine or not to mine?"):
    input_data = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
    
    # First detect if there's a mine
    is_mine = detection_model.predict(input_data)[0]
    
    if is_mine == 1:  # If a mine is detected
        # Then classify what type of mine it is
        mine_type = classification_model.predict(input_data)[0]
        
        mine_classes = {
            1: "AT Mine (Class 1)",
            2: "AP Mine (Class 2)",
            3: "Booby-trapped AP (Class 3)",
            4: "M14 AP (Class 4)"
        }
        
        st.success(f"Mine detected! Type: {mine_classes.get(mine_type, 'Unknown')}")
    else:
        st.success("No mine detected (Safe)")
