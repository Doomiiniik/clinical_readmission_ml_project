import pandas as pd
import joblib
import json
from src.pipeline import preprocess_data
from src.pipeline import MODEL_PATH
from src.preprocess import transform_with_scaler
SCALER_PATH = "models/scaler.pkl"

def predict_single(raw_record: dict):
    #load model
    model = joblib.load(MODEL_PATH)


    scaler = joblib.load(SCALER_PATH)
    df = pd.DataFrame([raw_record])
    X = preprocess_data(df)



    # load feature columns
    with open("models/feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    # add missing columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    # ensure correct order
    X = X[feature_cols]

    #sclaing record
    X = transform_with_scaler(X, scaler)
  
    # import best treshold
    with open("models/threshold.json") as f: 
        threshold = json.load(f)["threshold"] 

    proba = model.predict_proba(X)[:, 1][0]

    pred = int(proba >= threshold)
    

    return pred, proba
if __name__ == "__main__":
    df = pd.read_csv("data/raw/diabetic_data.csv")
    example = df.iloc[0].to_dict()

    pred, proba = predict_single(example)
    print("Prediction:", pred)
    print("Probability:", proba)

