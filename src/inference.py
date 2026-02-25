import pandas as pd
import joblib
import json
from src.pipeline import preprocess_data
from src.pipeline import MODEL_PATH

def predict_single(raw_record: dict):
    model = joblib.load(MODEL_PATH)

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

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    return pred, proba
if __name__ == "__main__":
    df = pd.read_csv("data/raw/diabetic_data.csv")
    example = df.iloc[0].to_dict()

    pred, proba = predict_single(example)
    print("Prediction:", pred)
    print("Probability:", proba)

