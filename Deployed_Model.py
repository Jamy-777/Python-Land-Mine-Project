import streamlit as st
import pandas as pd
import joblib

det_model = joblib.load("land_mines_randomforest_detection.joblib")
scaler = joblib.load("models/scaler_detection.joblib")

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
    raw = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
    
    raw[["V", "H"]] = scaler.transform(raw[["V", "H"]])
    
    for cat in [2, 3, 4, 5, 6]:
        raw[f"S_{cat}"] = (raw["S"] == cat).astype(int)
    X = raw.drop("S", axis=1)
    
    is_mine = det_model.predict(X)[0]
    mine_classes = {
        1: "Nuh uh (Class 1)",
        2: "AT Mine (Class 2)",
        3: "AP Mine (Class 3)",
        4: "Booby-trapped AP (Class 4)",
        5: "M14 AP (Class 5)"
    }
    st.success(f"Prediction: {mine_classes.get(is_mine, 'Unknown')}")
