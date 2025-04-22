import streamlit as st
import pandas as pd
import joblib

# Load both pipelines
det_model = joblib.load("land_mines_detection.joblib")
clf_model = joblib.load("land_mines_classification.joblib")

st.title("💣 Land Mine Detection App")
st.markdown("This app first detects if a mine is present, and if so, predicts its type.")

# User input
v = st.number_input("Voltage (V) [0 - 10.6 V]", min_value=0.0, max_value=10.6, step=0.1, value=5.5)
h = st.number_input("Height from ground (cm) [0 - 20 cm]", min_value=0.0, max_value=20.0, step=0.1, value=10.0)
s_type = st.selectbox("Soil Type", {
    1: "Dry and Sandy",
    2: "Dry and Humus",
    3: "Dry and Limy",
    4: "Humid and Sandy",
    5: "Humid and Humus",
    6: "Humid and Limy"
})

# Create input DataFrame
user_input = pd.DataFrame([[v, h, s_type]], columns=['V', 'H', 'S'])

# Run prediction
if st.button("Predict"):
    is_mine = det_model.predict(user_input)[0]

    if is_mine == 0:
        st.success("🟢 No mine detected. You're safe!")
    else:
        prediction = clf_model.predict(user_input)[0]
        mine_classes = {
            2: "AT Mine",
            3: "AP Mine",
            4: "Booby-trapped AP Mine",
            5: "M14 AP Mine"
        }
        result = mine_classes.get(prediction, "Unknown")
        st.error(f"🔴 Mine Detected! Type: **{result}** (Class {prediction})")
