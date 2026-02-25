# main.py
import pandas as pd
from src.data_loader import load_and_analyze, split_data_clinically
from src.preprocess import (
    
    scale_numeric_features,   
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

from src.train import  train_baseline, evaluate_model

def run_pipeline(data_path):
    # load the raw clinical dataset
    df = load_and_analyze(data_path)
    
    # execute the clinical preprocessing steps
   
    df = clean_missing_values(df)
    df = filter_eligible_population(df)
    df = binarize_target(df)
    df = engineer_features(df)
    df = encode_categorical_features(df)  
    

    No_Down_Stedy_Up_Columns= ['metformin','repaglinide','chlorpropamide','glimepiride'
         ,'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone'
         , 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',  'tolazamide'             
         , 'insulin', 'glyburide-metformin', 'glipizide-metformin'
         , 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'                                      
         ,'nateglinide']
   
    No_Yes_Ch_Columns = ['change', 'diabetesMed']

    df = encode_no_down_steady_up_as_dummies(df, No_Down_Stedy_Up_Columns)
    df = encode_binary_columns(df, No_Yes_Ch_Columns)
    
    df = map_icd_codes(df)

    df = pd.get_dummies(
    df,
    columns=['diag_1', 'diag_2', 'diag_3'],
    drop_first=True
    )

    df, _ = remove_rare_onehots(df)
    
    # split data using patient grouping to prevent identity leakage
    # patient number is the group identifier
    train_df, test_df = split_data_clinically(
        df,
        target_col="readmitted",
        group_col="patient_nbr"
    )
    
    
  
    # scale numerical columns based on training distribution
    train_df, test_df, scaler = scale_numeric_features(train_df, test_df)
    

    # define target and features
    y_train = train_df['target']
    y_test = test_df['target']
    x_train = train_df.drop(columns=['target', 'readmitted'])
    x_test = test_df.drop(columns=['target', 'readmitted'])
     # remove identifiers before training
    # they are not predictors of health
    x_train = x_train.drop(columns=['patient_nbr'])
    x_test = x_test.drop(columns=['patient_nbr'])
    




    # train the baseline logistic regression model
    # evaluate the results using clinical metrics
    model = train_baseline(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    
    return model

if __name__ == "__main__":
    # specify the path to your raw data
    path = "data/raw/diabetic_data.csv"
    trained_model = run_pipeline(path)