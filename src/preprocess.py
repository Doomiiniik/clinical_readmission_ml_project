# src/preprocess.py
from src.data_loader import load_and_analyze, split_data_clinically
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def clean_missing_values(df):
    # replace the question mark symbol with standard nan
    # this allows pandas to recognize missing entries
    df = df.replace(['?','None','none','nan'], np.nan)
    return df

def filter_eligible_population(df):
    # remove patients who died or are in hospice
    # these patients cannot be readmitted by definition
    # ids for expired or hospice care
    death_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(death_ids)]
    return df

def binarize_target(df):
    # create a binary classification target
    # positive case is readmission within thirty days
    df['target'] = (df['readmitted'] == '<30').astype(int)
    return df


def encode_categorical_features(df):
    # convert age ranges into numeric midpoints
    # age is often stored as buckets like ten to twenty
    age_dict = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, 
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df['age_numeric'] = df['age'].replace(age_dict).astype(int)
    
    # drop high cardinality or useless columns
    # weight and medical specialty have too many missing values
    # encounter id and patient number are just identifiers
    # also drop columns with only 'No' values ('citoglipton','examide')
    # as well as columns with very high percent of missing values:
    # 'max_glu_serum', 'A1Cresult', 'payer_code'
    cols_to_drop = ['age', 'weight', 'medical_specialty',
                     'payer_code', 'encounter_id', 'max_glu_serum','A1Cresult', 
                      'citoglipton','examide']
    df = df.drop(columns=cols_to_drop)
    
    # one hot encode remaining categorical variables
    # this creates binary columns for race and gender

    #inpute missing walues to Other or 
    df['race'] = df['race'].fillna('Other')
    df['gender'] = df['gender'].fillna('Unknown/Invalid')

    df = pd.get_dummies(df, columns=['race', 'gender'], drop_first=True)
    return df


def engineer_features(df):
    # combine previous hospital visits into a single score
    # high utilization is a strong predictor of risk
    df['total_utilization'] = (
        df['number_outpatient'] + 
        df['number_emergency'] + 
        df['number_inpatient']
    )

    df = df.drop(columns=[ 'number_outpatient', 'number_emergency', 'number_inpatient' ])

    return df


def scale_numeric_features(train_df, test_df):
    # scale numerical data to have zero mean and unit variance
    # this prevents large numbers from dominating the model
    scaler = RobustScaler()
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 
                    'total_utilization','num_procedures','num_medications'
                    ,'number_diagnoses','age_numeric']
    
   
   

    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    return train_df, test_df, scaler



from sklearn.preprocessing import RobustScaler
import joblib

NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "total_utilization",
    "age_numeric"
]

def fit_scaler_on_train(train_df):
    scaler = RobustScaler()
    scaler.fit(train_df[NUMERIC_COLS])
    return scaler

def transform_with_scaler(df, scaler):
    df_loc = df.copy()
    df_loc[NUMERIC_COLS] = scaler.transform(df_loc[NUMERIC_COLS])
    return df_loc



























def map_icd_codes(df):
    # create a mapping for clinical categories
    # codes are grouped by their numerical ranges
    # this reduces dimensionality and prevents overfitting
    
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    
    for col in diag_cols:
        # replace codes starting with letters to avoid errors
        # these are usually supplemental classifications
        df[col] = df[col].apply(lambda x: str(x) if x else 'other')
        
        # logic for grouping based on standard medical ranges
        # circulatory is four hundred to four hundred fifty nine
        # respiratory is four hundred sixty to five hundred nineteen
        # and so on for other systems
        def categorize(val):
            if 'V' in val or 'E' in val:
                return 'other'
            try:
                code = float(val)
                if 390 <= code <= 459 or code == 785:
                    return 'circulatory'
                elif 460 <= code <= 519 or code == 786:
                    return 'respiratory'
                elif 520 <= code <= 579 or code == 787:
                    return 'digestive'
                elif code == 250:
                    return 'diabetes'
                elif 800 <= code <= 999:
                    return 'injury'
                elif 710 <= code <= 739:
                    return 'musculoskeletal'
                elif 580 <= code <= 629 or code == 788:
                    return 'urogenital'
                elif 140 <= code <= 239:
                    return 'neoplasms'
                else:
                    return 'other'
            except ValueError:
                return 'other'
                
        df[col] = df[col].apply(categorize)
        
    
    return df

def remove_rare_onehots(df, rare_thresh=0.001, id_cols=None, drop_constant=True, save_path=None):
    # drop columns with very rare distinct values, below rare_tresh
    # id_cols -> columns than cant be delted
    
    if id_cols is None:
        id_cols = []

    df = df.copy()
    n = len(df)

    dropped = []

    # delete constant colums
    if drop_constant:
        constant_cols = [c for c in df.columns if c not in id_cols and df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            # saved into dropped
            dropped.extend(constant_cols)

    # detecting bool-like kolumns
    bool_like = []
    for c in df.columns:
        # ommit seelcted id columns
        if c in id_cols:
            continue
        # delete nan
        ser = df[c].dropna()
        # if empty after dropna -> ommit
        if ser.shape[0] == 0:
            continue
        uniques = set(ser.unique())

        # accept ONLY {0,1} or {True,False} or {'0','1'}
        if uniques.issubset({0,1}) or uniques.issubset({True,False}) or uniques.issubset({'0','1'}):
            bool_like.append(c)

    # count prevalence and drop columns with prevalence < rare treshold
    if bool_like:
        # convert to int detected columns
        safe_int = df[bool_like].replace({True:1, False:0, '1':1, '0':0}).fillna(0).astype(int)
        # percentage of distinct values
        prevalence = safe_int.sum(axis=0) / float(n)
        rare_cols = prevalence[prevalence < rare_thresh].index.tolist()
        if rare_cols:
            df.drop(columns=rare_cols, inplace=True)
            dropped.extend(rare_cols)

    # optional saving deleted columns
    if save_path and dropped:
        try:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({"dropped": dropped}, f, indent=2)
        except Exception:
            pass

    return df, dropped














'''

def encode_no_down_steady_up(df, columns):
    
    # map values: No -> 0, Down -> 1, Steady -> 2, Up -> 3
    
    mapping = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
    for col in columns:
        df[col] = df[col].map(mapping).fillna(0)
    return df
'''
def encode_no_down_steady_up_as_dummies(df, columns):
    
    # changes columns No/Down/Steady/Up na dummy variables (one-hot encoding).
    # safer than no->0 stedy->1 because we dont know the powe or theese relations
   
    for col in columns:
        # missing values in this categorical variables treated as diffrent category missing
        df[col] = df[col].fillna('Missing')
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df














def encode_binary_columns(df, columns):
    
    # map binary values:
    # No -> 0
    # Yes -> 1
    # Ch  -> 1
    # minning value -> -1
    
    mapping = {'No': 0, 'Yes': 1, 'Ch': 1}
    for col in columns:
        df[col] = df[col].map(mapping)
        df[col] = df[col].fillna(-1)
    return df




if __name__ == "__main__":
    
    df = load_and_analyze("./data/raw/diabetic_data.csv")
    
    print("=== Data loading ===")
    print(df.head())
    
    df = clean_missing_values(df)
    df = filter_eligible_population(df)
    df = binarize_target(df)
    df = engineer_features(df)
    df = encode_categorical_features(df)  
    df = map_icd_codes(df)

    No_Down_Stedy_Up_Columns= ['metformin','repaglinide','chlorpropamide','glimepiride'
         ,'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone'
         , 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',  'tolazamide'             
         , 'insulin', 'glyburide-metformin', 'glipizide-metformin'
         , 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'                                      
         ,'nateglinide']
   
    No_Yes_Ch_Columns = ['change', 'diabetesMed']

    df = encode_no_down_steady_up_as_dummies(df, No_Down_Stedy_Up_Columns)
    df = encode_binary_columns(df, No_Yes_Ch_Columns)
    df, _ = remove_rare_onehots(df)
    df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)
    
    










    train_df, test_df = split_data_clinically(
        df,
        target_col="readmitted",
        group_col="patient_nbr"
    )
    
    
    #train_df, test_df, scaler = scale_numeric_features(train_df, test_df)
    

    
    scaler = fit_scaler_on_train(train_df)
    train_df = transform_with_scaler(train_df, scaler)
    test_df = transform_with_scaler(test_df, scaler)

 
    joblib.dump(scaler, "./models/scaler.pkl")
    print("scaler saved at models/scaler.pkl")



    print("\n=== Summary after preprocessing ===")
    print(train_df.head())
    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)







    import os

    # ensure directory exists
    processed_path = "./data/processed"
    os.makedirs(processed_path, exist_ok=True)

    # save full preprocessed dataset before split
    df.to_csv(os.path.join(processed_path, "clinical_preprocessed_full.csv"), index=False)
    train_df.to_csv(os.path.join(processed_path, "train_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "test_preprocessed.csv"), index=False)

    print("\nPreprocessed dataset saved to data/processed/clinical_preprocessed_full.csv")
    print("Shape:", df.shape)