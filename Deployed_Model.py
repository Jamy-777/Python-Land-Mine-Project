import streamlit as st
import pandas as pd
import joblib

# Now I am loading my trained model and scaler here for creating the web app:

my_model = joblib.load("land_mines_logisticregression.joblib")
saved_scaler = joblib.load("land_mines_saved_scaler.joblib")

st.title("Land Mine Detection App (Hopefully ;) ")
st.markdown("This app predicts whether a land mine is present or not based on various sensory inputs, " \
"well I mean hopefully it does... It's not like you are gonna come complaining... right?")

v = st.number_input("Output voltage of FLC sensor due to magnetic distortion:", 
                    min_value=0.0, max_value=10.6, step=0.1, value=5.0)

h = st.number_input("Enter the height of the sensor from the ground:", min_value=0.0, max_value=20.0, step=0.5, value=10.0)
s = st.number_input("Soil type depending on moisture content:", min_value=1.0, max_value=6.0, step=1.0, value=3.0)

if st.button("To mine or not to mine?"):
    input_data = pd.DataFrame([[v, h, s]], columns=["V", "H", "S"])
    input_scaled = saved_scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=["V", "H", "S"])
    prediction = my_model.predict(input_scaled_df)[0]

    mine_classes = {
        1: "Nuh uh (Class 1)",
        2: "AT Mine (Class 2)",
        3: "AP Mine(Class 3)",
        4: "Booby-trapped AP (Class 4)",
        5: "M14 AP (Class 5)"
    }

    st.success(f" Prediction: {mine_classes.get(prediction, 'Unknown')}")
