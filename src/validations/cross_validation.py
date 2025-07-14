from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cross_validate_model(X ,y, model, n_splits=5, random_seed=42):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    confusion_matrices = []

    fold_metrics = []

    fold = 1
    
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # permutation = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_seed, scoring="accuracy")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        #importances = permutation.importances_mean
        
        print(f"\n{len(X_train[0])}")

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confusion_matrices.append(conf_matrix)

        #print(f"\nImportances:\n{importances}\n")
        
        fold_metrics.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
            #"feature_importance": importances
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
