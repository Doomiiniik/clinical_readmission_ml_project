## 📄 Hospital Readmission Prediction — Machine Learning Pipeline (Diabetes Dataset)
Overview

This project implements a complete machine learning pipeline for predicting 30‑day hospital readmission among diabetic patients. The goal is to build a clinically meaningful model: avoiding patient‑level leakage, ensuring proper preprocessing, tuning the decision threshold, interpreting model behavior, and validating the model using clinical metrics such as calibration and risk stratification.

The project is structured as a production‑style ML workflow with modular code, saved artifacts, reproducible transformations, and automated plot generation.
Dataset

The dataset is the Diabetes 130‑US hospitals dataset from UCI. It contains:

    over 100,000 hospital encounters

    demographic information

    diagnoses and procedures

    medications

    lab results

    readmission outcomes

The target variable is binary:

    1 → readmitted within 30 days

    0 → not readmitted

The dataset is imbalanced, with the positive class representing ~11% of cases.

## Pipeline Summary
Clinical Train/Test Split

The dataset is split by patient ID, ensuring that no patient appears in both training and test sets. This prevents leakage and mirrors real‑world deployment, where the model must generalize to unseen patients.
Preprocessing

The preprocessing pipeline includes:

    cleaning and normalizing data

    encoding categorical variables

    feature engineering

    scaling numerical features using RobustScaler

    saving the scaler and feature list for inference

All transformations are applied consistently during training and evaluation.

## Model Training

A Logistic Regression model is used as the baseline due to its interpretability and clinical acceptance.
Saved artifacts include:

    trained model

    scaler

    feature columns

    optimized threshold

##Threshold Optimization

Because the dataset is imbalanced, the default threshold of 0.5 performs poorly.
The project uses the precision‑recall curve to select a threshold that maximizes F1 for the positive class.

Final threshold: 0.135

This significantly improves recall for readmission cases.

## Model Performance
ROC Curve

The model achieves:

    ROC‑AUC ≈ 0.63

This is typical for this dataset and reflects the inherent difficulty of predicting readmission.
Precision, Recall, F1

With the optimized threshold:

    Precision ≈ 0.20

    Recall ≈ 0.35

    F1 ≈ 0.25

These values align with published baselines for this dataset.
Confusion Matrix

The optimized threshold improves detection of positive cases while maintaining reasonable precision.
Model Interpretation
Coefficients and Odds Ratios

Logistic Regression coefficients are extracted and converted to odds ratios to provide clinically interpretable insights.

Examples:

    Higher utilization, more diagnoses, and longer hospital stays increase readmission risk.

    Stable medication regimens and certain drug combinations are associated with lower risk.

A feature importance plot is saved in plots/feature_importance.png.
Calibration Analysis
Calibration Curve

The model is well‑calibrated in the low‑risk region (0.05–0.15) and slightly overestimates risk at higher predicted probabilities. This behavior is expected for Logistic Regression on imbalanced clinical data.

## Brier Score

Brier Score = 0.097

This indicates good probabilistic accuracy and is close to the theoretical baseline for an 11% prevalence dataset.

## Risk Stratification

Patients are divided into five equal‑sized groups (quintiles) based on predicted risk:

Risk Group	True Readmission Rate
Very Low	5.9%
Low	        7.9%
Medium	    10.6%
High	    12.0%
Very High	19.5%

This monotonic increase demonstrates that the model effectively sorts patients by clinical severity, even if absolute predictions are modest.

## Clinical Utility

A key question in real deployment is whether the model can identify a subset of patients who are meaningfully higher risk.

Using the top 20% highest‑risk predictions:

Readmission rate = 19.5%

This is almost 2× higher than the population baseline (~11%).
This makes the model useful for:

    targeted follow‑up

    discharge planning

    resource allocation

    early intervention programs

## Saved Plots

All plots are automatically saved to /plots:

    ROC Curve

    Precision‑Recall Curve

    Calibration Curve

    Feature Importance

    Additional clinical validation plots

These can be embedded into reports, presentations, or dashboards.


## Conclusions

This project demonstrates a complete, clinically grounded machine learning workflow:

    robust preprocessing

    leakage‑free patient‑level split

    interpretable baseline model

    threshold tuning

    calibration assessment

    risk stratification

    clinical utility evaluation

While the model is not highly predictive (ROC‑AUC ≈ 0.63), it is stable, interpretable, and can be clinically useful for identifying high‑risk patient groups.
