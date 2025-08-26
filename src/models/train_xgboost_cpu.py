import sys
import os
import numpy as np

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.validations.cross_validation import cross_validate_cpu
from src.models.model_utils import load_features

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..",  "data/interim/")


def train_xgboost_cv_cpu(dataset_name, n_splits=5):


    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = XGBClassifier(
        objective="multi:softrprob",
        random_state=42
    )
    
    results, fold_metrics = cross_validate_cpu(X, y_encoded, model, n_splits=n_splits, random_seed=42)

    return 



def train_xgboost_cpu(dataset_name, test_size=0.3):


    dataset_filename = DATASET_PATH + dataset_name

    X, y = load_features(dataset_filename)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = XGBClassifier(
        objective="multi:softrprob",
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, shuffle=True, stratify=y, random_state=42)

    print(f"\nTraining XGBClassifier on CPU on dataset {dataset_name} using a test_size of {test_size*100}%.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== XGBClassifier evaluation on {test_size*100}% test size ===")
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
        pass
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        dataset_filename = dataset_filename + ".npz"
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit

    option = 0

    while option < 1 or option > 3:

        option = int (input("\nWhat type of training do you want to do:" \
        "\n1. Cross validation" \
        "\n2. Single training" \
        "\n3. Hyperparameter tunning" \
        "\n\nOption: "))

    if option == 1:
        train_xgboost_cv_cpu(dataset_filename)
    elif option == 2:
        train_xgboost_cpu(dataset_filename)
    elif option == 3:
        pass