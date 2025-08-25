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

def train_weighted_boosting(dataset_name, n_splits=5):

    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y, random_state=42)

    
    xgb = XGBClassifier(
        objective="multi:softrprob",
        random_state=42
        )
    
    lgbm = LGBMClassifier(
        objective="multiclass", 
        verbosity=0,
        random_state=42
        )

    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    xgb_probs = xgb.predict_proba(X_test)  
    lgbm_probs = lgbm.predict_proba(X_test)


    weight_xgb = 0.5
    weight_lgbm = 0.5

    ensemble_props = weight_xgb * xgb_probs + weight_lgbm * lgbm_probs

    y_pred = np.argmax(ensemble_props, axis=1)

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
        train_weighted_boosting(dataset_filename)
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        train_weighted_boosting(dataset_filename + ".npz")
        
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit
