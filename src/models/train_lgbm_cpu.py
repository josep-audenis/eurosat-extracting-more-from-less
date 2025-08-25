import sys
import os
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.validations.cross_validation import cross_validate_cpu
from src.models.model_utils import load_features

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..",  "data/interim/")


def train_lgbm_cv_cpu(dataset_name, n_splits=5):


    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = LGBMClassifier(
        objective="multiclass", 
        verbosity=0,
        random_state=42)
    
    results, fold_metrics = cross_validate_cpu(X, y, model, n_splits=n_splits, random_seed=42)

    # Saving model ???




if __name__ == "__main__":
    print("Available datasets:")
    for dataset in os.listdir(DATASET_PATH):
        print(f"- {dataset}")
    
    dataset_filename = input("What dataset would you like to use: ")

    if dataset_filename in os.listdir(DATASET_PATH):
        train_lgbm_cv_cpu(dataset_filename)
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        train_lgbm_cv_cpu(dataset_filename + ".npz")
        
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit