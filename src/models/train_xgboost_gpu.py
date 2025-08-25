import sys
import os
import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.validations.cross_validation_gpu import cross_validate_xgb_gpu
from src.models.model_utils import load_features

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..",  "data/interim/")


def train_xgboost_cv(dataset_name, n_splits=5):


    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_encoded)),
        "random_state": 42,
        "device": "cuda"
    }
    
    results, fold_metrics = cross_validate_xgb_gpu(X, y_encoded, params, n_splits=n_splits, random_seed=42)

    # Saving model ???




if __name__ == "__main__":
    dataset_filename = input("What dataset would you like to use: ")

    if dataset_filename in os.listdir(DATASET_PATH):
        train_xgboost_cv(dataset_filename)
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        train_xgboost_cv(dataset_filename + ".npz")
        
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit