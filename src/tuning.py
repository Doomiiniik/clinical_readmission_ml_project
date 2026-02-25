# src/tuning.py
import json
import warnings
from pathlib import Path
import time

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

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

# configuration
DATA_PATH = "data/raw/diabetic_data.csv"
N_TRIALS = 60
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_PATH = Path("models/best_params.json")
BEST_MODEL_PATH = Path("models/best_model.joblib")
LOG_TRIALS_PATH = Path("models/optuna_trials.jsonl")

# no annoying warinings about deprecated functions
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# helpers

def choose_remove_fn():
 
        return remove_rare_onehots

def prepare_df(path: str) -> pd.DataFrame:
 #Load and run preprocessing pipeline up to the point 
 # where we have a feature matrix.
    df = load_and_analyze(path)

    df = clean_missing_values(df)
    df = filter_eligible_population(df)
    df = binarize_target(df)
    df = engineer_features(df)  
    df = encode_categorical_features(df)

    No_Down_Steady_Up_Columns = [
        'metformin','repaglinide','chlorpropamide','glimepiride',
        'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
        'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
        'insulin','glyburide-metformin','glipizide-metformin',
        'glimepiride-pioglitazone','metformin-rosiglitazone',
        'metformin-pioglitazone','nateglinide'
    ]
    No_Yes_Ch_Columns = ['change', 'diabetesMed']

    df = encode_no_down_steady_up_as_dummies(df, No_Down_Steady_Up_Columns)
    df = encode_binary_columns(df, No_Yes_Ch_Columns)

    df = map_icd_codes(df)

    # one-hot diag columns
    df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)

    # remove rare one-hots 
    remove_fn = choose_remove_fn()
    try:
        df, dropped = remove_fn(df, rare_thresh=0.001, id_cols=['patient_nbr'], drop_constant=True)
    except TypeError:
        # fallback if signature differs
        df, dropped = remove_fn(df)
    print(f"Removed {len(dropped)} rare/constant columns")
    return df
###############################################################

def build_preprocessor(X: pd.DataFrame):
    """Return ColumnTransformer that scales numeric columns only (dense path)."""
    exclude = {'target', 'readmitted', 'patient_nbr', 'encounter_id'}
    numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols)
    ], remainder="passthrough", sparse_threshold=0)

    return preprocessor, numeric_cols


def to_sparse_if_needed(X: pd.DataFrame, threshold: int = 800):
    """Convert to CSR sparse matrix if number of columns exceeds threshold."""
    if X.shape[1] > threshold:
        return sp.csr_matrix(X.values)
    return X


def fit_and_score_model(model, X_train, y_train, X_val, y_val):
    """Fit model and return AUC. Handles predict_proba/decision_function."""
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_val)[:, 1]
    else:
        preds = model.decision_function(X_val)
    return roc_auc_score(y_val, preds)


def objective(trial: optuna.trial.Trial, X, y, groups):
    """
    Optuna objective: GroupKFold CV and mean AUC.
    - ensures l1_ratio only used with elasticnet
    - uses saga solver for LogisticRegression
    - uses early_stopping for SGDClassifier
    """
    model_choice = trial.suggest_categorical("model", ["logreg", "sgd"])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    max_iter = trial.suggest_int("max_iter", 500, 3000)
    tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)

    # regularization params
    l1_ratio = None
    if model_choice == "logreg":
        C = trial.suggest_loguniform("C", 1e-3, 1e2)
        solver = "saga"  # saga supports l1/l2/elasticnet
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
    else:
        alpha = trial.suggest_loguniform("alpha", 1e-6, 1e-1)
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)

    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs = []
    use_sparse = sp.issparse(X)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        # split
        if use_sparse:
            X_train = X[train_idx]
            X_val = X[val_idx]
        else:
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # scaling
        if use_sparse:
            scaler = MaxAbsScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # build model with consistent params
        if model_choice == "logreg":
            # pass l1_ratio only when penalty == 'elasticnet'
            if penalty == "elasticnet":
                model = LogisticRegression(
                    C=C,
                    penalty=penalty,
                    l1_ratio=l1_ratio,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            else:
                model = LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
        else:
            # SGDClassifier with early stopping to avoid long non-converging runs
            if penalty == "elasticnet":
                model = SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    tol=tol,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    average=True,
                    random_state=RANDOM_STATE
                )
            else:
                model = SGDClassifier(
                    loss="log_loss",
                    penalty=penalty,
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    average=True,
                    random_state=RANDOM_STATE
                )

        # fit and score
        auc = fit_and_score_model(model, X_train, y_train, X_val, y_val)
        aucs.append(auc)

        # report and pruning
        trial.report(np.mean(aucs), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(aucs))


def run_optuna_tuning(path: str, n_trials: int = N_TRIALS):
    df = prepare_df(path)

    # split clinically and use only train for tuning
    train_df, _ = split_data_clinically(df, target_col="readmitted", group_col="patient_nbr")

    # scale numeric features in-place to match pipeline (no leakage)
    train_df, _, _ = scale_numeric_features(train_df, train_df)

    y = train_df['target']
    X = train_df.drop(columns=['target', 'readmitted', 'patient_nbr'], errors='ignore')
    groups = train_df['patient_nbr']

    # build preprocessor for dense path and apply
    preprocessor, numeric_cols = build_preprocessor(X)
    X_dense = preprocessor.fit_transform(X)
    X_proc = to_sparse_if_needed(pd.DataFrame(X_dense), threshold=800)

    # create study with pruning and logging callback
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))

    def log_trial(study, trial):
        rec = {"trial_number": trial.number, "params": trial.params, "value": float(trial.value) if trial.value is not None else None, "state": str(trial.state)}
        LOG_TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_TRIALS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    try:
        study.optimize(lambda t: objective(t, X_proc, y, groups), n_trials=n_trials, callbacks=[log_trial], show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optuna tuning interrupted by user. Saving best found so far...")

    best = study.best_params
    best_value = study.best_value
    result = {"best_params": best, "best_cv_auc": best_value}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # train final model on full train_df with best params
    final_model_choice = best.get("model", "logreg")
    final_penalty = best.get("penalty", "l2")
    final_max_iter = best.get("max_iter", 1000)
    final_tol = best.get("tol", 1e-4)

    if final_model_choice == "logreg":
        final_C = best.get("C", 1.0)
        if final_penalty == "elasticnet":
            final_l1_ratio = best.get("l1_ratio", 0.15)
            final_clf = LogisticRegression(C=final_C, penalty=final_penalty, l1_ratio=final_l1_ratio, solver="saga", max_iter=final_max_iter, tol=final_tol, class_weight=best.get("class_weight", None), n_jobs=-1, random_state=RANDOM_STATE)
        else:
            final_clf = LogisticRegression(C=final_C, penalty=final_penalty, solver="saga", max_iter=final_max_iter, tol=final_tol, class_weight=best.get("class_weight", None), n_jobs=-1, random_state=RANDOM_STATE)
    else:
        final_alpha = best.get("alpha", 1e-4)
        if final_penalty == "elasticnet":
            final_clf = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=final_alpha, l1_ratio=best.get("l1_ratio", 0.15), max_iter=final_max_iter, tol=final_tol, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, average=True, random_state=RANDOM_STATE)
        else:
            final_clf = SGDClassifier(loss="log_loss", penalty=final_penalty, alpha=final_alpha, max_iter=final_max_iter, tol=final_tol, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, average=True, random_state=RANDOM_STATE)

    # final scaling and fit on full train
    use_sparse_final = sp.issparse(X_proc)
    if use_sparse_final:
        scaler = MaxAbsScaler()
        X_full = scaler.fit_transform(X_proc)
    else:
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X)

    final_clf.fit(X_full, y)

    # save final model components (scaler + classifier)
    artifacts_dir = BEST_MODEL_PATH.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": final_clf, "preprocessor": preprocessor}, BEST_MODEL_PATH)

    print("=== Optuna finished ===")
    print(json.dumps(result, indent=2))
    print("Saved final model to", BEST_MODEL_PATH)
    return result


if __name__ == "__main__":
    start = time.time()
    run_optuna_tuning(DATA_PATH, n_trials=N_TRIALS)
    print("Total tuning time (s):", round(time.time() - start, 1))
