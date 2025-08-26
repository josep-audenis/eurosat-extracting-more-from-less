import time
import os
import psutil
import joblib

import numpy as np

from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_features(feature_file):
    data = np.load(feature_file)
    return data['X'], data['y']

def evaluate_model(model, X_train, X_test, y_train, y_test):
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error")
    tracker.start()

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    emissions = tracker.stop()

    y_pred = model.predict(X_test)

    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2) 

    joblib.dump(model, "temp_model.pkl")
    model_size = os.path.getsize("temp_model.pkl") / (1024 ** 2)
    os.remove("temp_model.pkl")

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average="macro"), recall_score(y_test, y_pred, average="macro"), f1_score(y_test, y_pred, average="macro"), train_time, mem_usage, model_size, emissions