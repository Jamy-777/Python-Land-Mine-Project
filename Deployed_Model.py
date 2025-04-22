import streamlit as st
import pandas as pd
import joblib

# load your full pipeline
my_model = joblib.load("land_mines_randomforest_detection.joblib")

st.title("Land Mine Detection App (Hopefully ;)")
st.markdown("…")

soil_type_values = {
    1: "Dry and Sandy",
    2: "Dry and Humus",
    3: "Dry and Limy",
    4: "Humid and Sandy",
    5: "Humid and Humus",
    6: "Humid and Limy"
}

v = st.number_input("Output voltage …", 0.0, 10.6, 5.0, step=0.1)
h = st.number_input("Enter the height … (cm):", 0.0, 20.0, 10.0, step=0.5)

s = st.selectbox(
    "Soil type:",
    options=[1,2,3,4,5,6],
    format_func=lambda i: soil_type_values[i]
)

if st.button("To mine or not to mine?"):
    input_data = pd.DataFrame([[v, h, s]], columns=["V","H","S"])
    prediction = my_model.predict(input_data)[0]

    mine_classes = {
        1: "Nuh uh (Class 1)",
        2: "AT Mine (Class 2)",
        3: "AP Mine (Class 3)",
        4: "Booby-trapped AP (Class 4)",
        5: "M14 AP (Class 5)"
    }

    st.success(f"Prediction: {mine_classes.get(prediction, 'Unknown')}")
