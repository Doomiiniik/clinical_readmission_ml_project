# src/train.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def train_baseline(x_train, y_train):
    # initialize the logistic regression model
    # use balanced weights to handle the rare target class
    # increase max iterations to ensure convergence
    model = LogisticRegression(
        class_weight='balanced', 
        max_iter=5000, 
        random_state=42
    )
    
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    # generate predictions and probabilities
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    
    # print metrics that matter for clinical tasks
    # recall is more important than precision here
    print("classification report:")
    print(classification_report(y_test, preds))
    
    # auc roc is standard but pr auc is better for imbalance
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    print(f"auc roc: {roc_auc:.4f}")
    print(f"pr auc: {pr_auc:.4f}")