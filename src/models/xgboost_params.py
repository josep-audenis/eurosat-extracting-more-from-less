XGBOOST_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
    "n_estimators": [100, 300, 500],
}