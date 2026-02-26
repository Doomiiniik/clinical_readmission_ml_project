# src/pipeline.py
# clean, stable pipeline builder with no joblib pickling of classes

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from src.preprocess import (
    clean_missing_values,
    filter_eligible_population,
    binarize_target,
    engineer_features,
    encode_categorical_features,
    map_icd_codes,
    encode_no_down_steady_up_as_dummies,
    encode_binary_columns,
    remove_rare_onehots
)

BEST_PARAMS_PATH = Path("models/best_params.json")
MODEL_PATH = Path("models/model.joblib")


def preprocess_data(df):
    # exact preprocessing steps from main.py

    df = clean_missing_values(df)
    df = filter_eligible_population(df)
    df = binarize_target(df)

    df = engineer_features(df)
    df = encode_categorical_features(df)

    no_down_cols = [
        'metformin','repaglinide','chlorpropamide','glimepiride','acetohexamide',
        'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone',
        'acarbose','miglitol','troglitazone','tolazamide','insulin',
        'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
        'metformin-rosiglitazone','metformin-pioglitazone','nateglinide'
    ]
    df = encode_no_down_steady_up_as_dummies(df, no_down_cols)

    df = encode_binary_columns(df, ['change', 'diabetesMed'])
    df = map_icd_codes(df)

    df = pd.get_dummies(df, columns=['diag_1','diag_2','diag_3'], drop_first=True)

    df, _ = remove_rare_onehots(df)

    return df


def build_model():
    #create a logistic regression model from best_params.json

    with open(BEST_PARAMS_PATH, "r") as f:
        params = json.load(f)["best_params"]

    params.pop("model", None)  # remove non-logreg key
    
    model = LogisticRegression(solver="liblinear", **params)

    return model


def build_pipeline():
    
    #    returns a tuple (preprocess_function, model)
    #    inference will call:
    #    X = preprocess_function(df)
    #    y = model.predict(X)
    

    model = build_model()

    # if trained model exists, load it
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    return preprocess_data, model
