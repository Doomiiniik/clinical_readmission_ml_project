import pandas as pd
import joblib
from src.pipeline import preprocess_data, build_model
from src.data_loader import split_data_clinically

RAW_PATH = "data/raw/diabetic_data.csv"
MODEL_PATH = "models/model.joblib"

def train_and_save_model():
    # load raw data
    df = pd.read_csv(RAW_PATH)

    # full preprocessing
    df = preprocess_data(df)

    # split into train/test using your clinical grouping
    train_df, test_df = split_data_clinically(
        df,
        target_col="target",
        group_col="patient_nbr"
    )

    # training data
    y_train = train_df["target"]
    X_train = train_df.drop(columns=["target", "readmitted", "patient_nbr", "encounter_id"], errors="ignore")

    # build model with best_params.json
    model = build_model()

    # train ONLY on train
    model.fit(X_train, y_train)

    import json

    feature_cols = X_train.columns.tolist()

    with open("models/feature_columns.json", "w") as f:
     json.dump(feature_cols, f)

    # save trained model
    joblib.dump(model, MODEL_PATH)
    print("Model trained on TRAIN and saved to", MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
