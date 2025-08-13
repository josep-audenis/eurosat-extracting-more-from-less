import numpy as np
import cupy as cp
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def cross_validate_xgb_gpu(X, y, params, num_boost_round=100, n_splits=5, random_seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    accuracies, precisions, recalls, f1s, confusion_matrices = [], [], [], [], []
    fold_metrics = []
    fold = 1
    
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)
    if not isinstance(y, cp.ndarray):
        y = cp.asarray(y)

    for train_idx, test_idx in skf.split(cp.asnumpy(X), cp.asnumpy(y)):  
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        
        y_pred_prob = booster.predict(dtest)
        
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        y_test_np = cp.asnumpy(y_test)

        accuracy = accuracy_score(y_test_np, y_pred)
        precision = precision_score(y_test_np, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test_np, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test_np, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_test_np, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confusion_matrices.append(conf_matrix)

        fold_metrics.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
        })

        print(f"Fold {fold}:\n\tAccuracy={accuracy*100:.2f}%\n\tPrecision={precision*100:.2f}%\n\tRecall={recall*100:.2f}%\n\tF1={f1*100:.2f}%\n")
        fold += 1

    results = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s)
    }
    
    print("\n=== Cross-Validation Summary ===")
    print(f"Accuracy: {results['accuracy_mean']*100:.2f}% ± {results['accuracy_std']*100:.2f}%")
    print(f"Precision (Macro): {results['precision_mean']*100:.2f}% ± {results['precision_std']*100:.2f}%")
    print(f"Recall (Macro): {results['recall_mean']*100:.2f}% ± {results['recall_std']*100:.2f}%")
    print(f"F1 Score (Macro): {results['f1_mean']*100:.2f}% ± {results['f1_std']*100:.2f}%\n")
    

    return results, fold_metrics
