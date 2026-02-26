import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

from src.pipeline import preprocess_data
from src.data_loader import split_data_clinically
from src.preprocess import transform_with_scaler
RAW_PATH = "data/raw/diabetic_data.csv"
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.pkl"

def evaluate_model():
    
    df = pd.read_csv(RAW_PATH)

    
    df = preprocess_data(df)
    
    _ , test_df = split_data_clinically(
        df,
        target_col="target",
        group_col="patient_nbr"
    )

    
    
    
    scaler = joblib.load(SCALER_PATH)
    test_df = transform_with_scaler(test_df, scaler)

       


  
    y_test = test_df["target"]
    # drop columns connected with target to dont allow data leakage
    X_test = test_df.drop(columns=["target", "readmitted", "patient_nbr", "encounter_id"], errors="ignore")



    
    model = joblib.load(MODEL_PATH)
    with open("models/feature_columns.json", "r") as f:
     feature_cols = json.load(f)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    
    
    
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # F1 for each treshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    

    

    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold)}, f)

    print(f"\nSaved threshold to models/threshold.json: {best_threshold}")


    
    print("\n=== Best threshold (max F1) ===")
    print(best_threshold)
    print("Precision:", precision[best_idx])
    print("Recall:", recall[best_idx])
    print("F1:", f1_scores[best_idx])


    y_pred_thresh = (y_proba >= best_threshold).astype(int)
    


















    print("\n=== ROC-AUC ===")
    print(roc_auc_score(y_test, y_proba))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_thresh))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred_thresh))

    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
