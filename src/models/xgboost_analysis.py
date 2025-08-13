import sys
import os
import json
import numpy as np
import cupy as cp
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.xgboost_params import XGBOOST_GRID
from src.models.model_utils import load_features

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../..", "data/interim/")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../..", "data/processed/")

def parameter_optimization_xgb_gpu_native(dataset_name, n_splits_outer=5, n_splits_inner=3, n_iter=15):
    dataset_filename = os.path.join(DATASET_PATH, dataset_name)

    X, y = load_features(dataset_filename)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

    all_results = {
        "dataset": dataset_name,
        "folds": [],
        "summary": {}
    }

    fold_num = 1
    for train_idx, test_idx in outer_cv.split(X, y_encoded):
        print(f"\n=== Outer Fold {fold_num} ===")

        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        param_list = list(ParameterSampler(XGBOOST_GRID, n_iter=n_iter, random_state=42))
        best_params = None
        best_inner_score = -np.inf

        for params in param_list:
            params = params.copy()
            params.update({
                "objective": "multi:softprob",
                "num_class": len(np.unique(y_encoded)),
                "device": "cuda",
                "verbosity": 0,
            })

            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner, X_val_inner = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
                y_train_inner, y_val_inner = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

                dtrain_inner = xgb.DMatrix(cp.asarray(X_train_inner), label=cp.asarray(y_train_inner))
                dval_inner = xgb.DMatrix(cp.asarray(X_val_inner), label=cp.asarray(y_val_inner))

                booster_inner = xgb.train(params, dtrain_inner, num_boost_round=100, verbose_eval=False)
                y_pred_prob_inner = booster_inner.predict(dval_inner)
                y_pred_inner = cp.asnumpy(cp.argmax(cp.asarray(y_pred_prob_inner), axis=1))
                inner_scores.append(accuracy_score(y_val_inner, y_pred_inner))

            mean_inner_score = np.mean(inner_scores)
            if mean_inner_score > best_inner_score:
                best_inner_score = mean_inner_score
                best_params = params

        print(f"Best Params (from inner CV) for Fold {fold_num}: {best_params}")

        dtrain_outer = xgb.DMatrix(cp.asarray(X_train_outer), label=cp.asarray(y_train_outer))
        dtest_outer = xgb.DMatrix(cp.asarray(X_test_outer), label=cp.asarray(y_test_outer))

        booster_outer = xgb.train(best_params, dtrain_outer, num_boost_round=100, verbose_eval=False)
        y_pred_prob_outer = booster_outer.predict(dtest_outer)
        y_pred_outer = cp.asnumpy(cp.argmax(cp.asarray(y_pred_prob_outer), axis=1))

        accuracy = accuracy_score(y_test_outer, y_pred_outer)
        precision = precision_score(y_test_outer, y_pred_outer, average="macro", zero_division=0)
        recall = recall_score(y_test_outer, y_pred_outer, average="macro", zero_division=0)
        f1 = f1_score(y_test_outer, y_pred_outer, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_test_outer, y_pred_outer).tolist()

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": conf_matrix
        }

        all_results["folds"].append({
            "fold": fold_num,
            "metrics": metrics,
            "best_params": best_params
        })

        print(f"Fold {fold_num} Accuracy: {accuracy:.4f}")
        fold_num += 1

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [fold["metrics"][metric] for fold in all_results["folds"]]
        all_results["summary"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    print("\n=== Final Nested CV Results ===")
    print(f"Mean Accuracy: {all_results['summary']['accuracy']['mean']:.4f} ± {all_results['summary']['accuracy']['std']:.4f}")

    dataset_name = dataset_name.split(".")[0]
    with open(os.path.join(RESULTS_PATH, f"{dataset_name}_xgboost_analysis.json"), "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    dataset_filename = input("What dataset would you like to use: ")

    if dataset_filename in os.listdir(DATASET_PATH):
        parameter_optimization_xgb_gpu_native(dataset_filename)
    elif dataset_filename + ".npz" in os.listdir(DATASET_PATH):
        parameter_optimization_xgb_gpu_native(dataset_filename + ".npz")
    else:
        print(f"No {dataset_filename} or {dataset_filename}.npz found in {DATASET_PATH}")
        exit()
