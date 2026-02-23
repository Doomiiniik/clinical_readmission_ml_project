# src/data_loader.py
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_and_analyze(file_path):
    # load the raw clinical data
    df = pd.read_csv(file_path)
    
    # identify the patient identifier column
    # in this dataset it is patient_nbr
    patient_col = 'patient_nbr'
    
    # check for repeat visitors
    counts = df[patient_col].value_counts()
    repeat_patients = counts[counts > 1].count()
    
    print(f"total records: {len(df)}")
    print(f"unique patients: {df[patient_col].nunique()}")
    print(f"patients with multiple visits: {repeat_patients}")
    
    return df

def split_data_clinically(df, target_col, group_col):
    # perform a split that respects patient identity
    # this prevents the model from memorizing specific patients
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # get the indices for the split
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    return train_df, test_df



if __name__ == "__main__":
    
    df = load_and_analyze("./data/raw/diabetic_data.csv")
    
    train_df, test_df = split_data_clinically(
        df,
        target_col="readmitted",
        group_col="patient_nbr"
    )
    
    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)