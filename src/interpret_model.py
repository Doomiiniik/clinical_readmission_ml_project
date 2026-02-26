import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/model.joblib"
FEATURES_PATH = "models/feature_columns.json"

def interpret_model():

    # load model
    model = joblib.load(MODEL_PATH)

    # load feature names
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
        "odds_ratio": np.exp(coefs)
    }).sort_values("abs_coef", ascending=False)

    print("\n=== Top features that increase risk (coef > 0) ===")
    print(df[df["coef"] > 0].head(20)[["feature", "coef", "odds_ratio"]])

    print("\n=== Top features that reduce risk (coef < 0) ===")
    print(df[df["coef"] < 0].head(20)[["feature", "coef", "odds_ratio"]])

    
    top_n = 20
    top_features = df.head(top_n).sort_values("coef")

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["coef"])
    plt.xlabel(" coef")
    plt.title("Top 20 most important features (Logistic Regression)")
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    interpret_model()
