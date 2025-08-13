import sys
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.xgboost_params import XGBOOST_GRID

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..",  "data/interim/")

if __name__ == "__main__":
    dataset_filename = input("Waht dataset would you like to use: ")

    if dataset_filename in DATASET_PATH:
        print(".npz")
    elif dataset_filename + "npz" in DATASET_PATH:
        print("no .npz")
    else:
        print(f"No {dataset_filename}  or {dataset_filename}.npz found in {DATASET_PATH}")
        exit