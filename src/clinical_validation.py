import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.pipeline import preprocess_data
from src.data_loader import split_data_clinically
from src.preprocess import transform_with_scaler

RAW_PATH = "data/raw/diabetic_data.csv"
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.json"
THRESHOLD_PATH = "models/threshold.json"


def clinical_validation():

    print("=== Starting clinical validation ===")

    # load and preprocess raw data
    df = pd.read_csv(RAW_PATH)
    df = preprocess_data(df)

    # clinical split (ensures no patient leakage)
    _, test_df = split_data_clinically(df, target_col="target", group_col="patient_nbr")

    # load scaler and apply it to test data
    scaler = joblib.load(SCALER_PATH)
    test_df = transform_with_scaler(test_df, scaler)

    # prepare X and y
    y_test = test_df["target"]
    X_test = test_df.drop(columns=["target", "readmitted", "patient_nbr", "encounter_id"], errors="ignore")

    # load trained model
    model = joblib.load(MODEL_PATH)

    # load feature columns and reorder X_test
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    X_test = X_test[feature_cols]

    # Load threshold
    with open(THRESHOLD_PATH, "r") as f:
        threshold = json.load(f)["threshold"]

    # predict probabilities and thresholded predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_thresh = (y_proba >= threshold).astype(int)

    

    # ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # === Calibration curve ===
    print("\n=== Calibration Curve ===")
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve")
    plt.legend()

    calibration_path = "plots/calibration_curve.png"
    plt.savefig(calibration_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration curve saved to: {calibration_path}")

    # === Brier score ===
    brier = brier_score_loss(y_test, y_proba)
    print("\n=== Brier Score ===")
    print(brier)

    # === Risk stratification ===
    print("\n=== Risk Stratification ===")
    test_df["proba"] = y_proba
    test_df["risk_group"] = pd.qcut(
        test_df["proba"], 
        q=5, 
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    risk_table = test_df.groupby("risk_group")["target"].mean()
    count_table = test_df.groupby("risk_group")["target"].count()

    print("\nAverage readmission rate per risk group:")
    print(risk_table)

    print("\nNumber of patients per group:")
    print(count_table)

    # === Clinical utility ===
    print("\n=== Clinical Utility ===")
    top_20 = test_df[test_df["proba"] >= test_df["proba"].quantile(0.80)]
    recall_top20 = top_20["target"].mean()
    print(f"Readmission rate in top 20% highest-risk patients: {recall_top20:.3f}")

    # === Precision-Recall curve ===
    print("\n=== Precision-Recall Curve ===")
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    pr_path = "plots/precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Precision-Recall curve saved to: {pr_path}")

if __name__ == "__main__":
    clinical_validation()
