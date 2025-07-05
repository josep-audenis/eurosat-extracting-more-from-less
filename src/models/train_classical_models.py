import os
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_features(feature_file):
    data = np.load(feature_file)
    return data['X'], data['y']

def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

def train_xgboost(X_train, y_train):
    return

def train_classic(model="random_forest", train_ratio=None, cross_validation=None, dataset=None, model_filename="random_forest_model"):
    
    if dataset is None:
        print("Error in train_classic: dataset field is None")
        return
    
    dataset_filename = dataset + ".npz"

    if dataset_filename not in os.listfiles("/data/interim/"):
        print(f"Error in train_classic: dataset file {dataset_filename} not found.")
        return
    
    X, y = load_features("/data/interim/" + dataset_filename)
    
    if cross_validation is not None: # cross_validation

    else:

    

if __name__ == "__main__":
    X, y = load_features("data/interim/features_train.npz")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    rfc = train_random_forest(X_train, y_train)

    y_pred = rfc.predict(X_val)

    dump(rfc, "models/classic_model.joblib")
