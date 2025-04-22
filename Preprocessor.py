import pandas as pd
import joblib

scaler = joblib.load("land_mines_randomforest_detection.joblib")  

SOIL_CATS = [1, 2, 3, 4, 5, 6]

def preprocess_for_detection(df: pd.DataFrame) -> pd.DataFrame:
    df[['V','H']] = scaler.transform(df[['V','H']])
    
    for cat in SOIL_CATS[1:]:  
        df[f"S_{cat}"] = (df["S"] == cat).astype(int)
    
    return df.drop("S", axis=1)