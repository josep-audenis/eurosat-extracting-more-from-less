import os
import sys

import numpy as np

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.model_utils import load_features

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..",  "data/interim/")

def get_oof_preds(X, y, models, n_splits=5, random_state=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y))
    oof_preds = np.zeros((X.shape[0], len(models) * n_classes))

    for i, model  in enumerate(models):
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)  # shape: (val_size, n_classes)

            print(f"{model} - {preds}")

            start_col = i * n_classes
            end_col = (i + 1) * n_classes
            oof_preds[val_idx, start_col:end_col] = preds

    return oof_preds

def train_oof(dataset_name, n_splits=5):

    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y, random_state=42)

    rf = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1
        )
    
    xgb = XGBClassifier(
        objective="multi:softrprob",
        random_state=42
        )
    
    lgbm = LGBMClassifier(
        objective="multiclass", 
        verbosity=0,
        random_state=42
        )

    base_models = [
        rf, 
        xgb, 
        lgbm]

    oof_train = get_oof_preds(X_train, y_train, base_models, n_splits=n_splits)

    meta_model = LogisticRegression(multi_class="multinomial")
    meta_model.fit(oof_train, y_train)

    n_classes = len(np.unique(y))
    test_preds = np.zeros((X_test.shape[0], len(base_models) * n_classes))

    for i, model in enumerate(base_models):
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        start_col = i * n_classes
        end_col = (i + 1) * n_classes
        test_preds[:, start_col:end_col] = preds

    y_pred = meta_model.predict(test_preds)

    print("\n=== Meta-Model Evaluation on Hold-Out Test ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')*100:.2f}%")
    print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')*100:.2f}%")
    print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro')*100:.2f}%\n")

    return



if __name__ == "__main__":
    print("Available datasets:")
    for dataset in os.listdir(DATASET_PATH):
        print(f"- {dataset}")

    dataset_filename = input("What dataset would you like to use: ")

    if dataset_filename in os.listdir(DATASET_PATH):
        train_oof(dataset_filename)
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        train_oof(dataset_filename + ".npz")
        
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit
