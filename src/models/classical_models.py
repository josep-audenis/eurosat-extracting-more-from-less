import os
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from validations.cross_validation import cross_validate_model
from reports.report_generator import generate_cross_validation_report


def load_features(feature_file):
    data = np.load(feature_file)
    return data['X'], data['y']



def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(objective="multi:softprob", random_state=42)
    xgb.fit(X_train, y_train)
    return xgb



def train_classic(model_name="random_forest", train_ratio=None, cross_validation=None, dataset=None, model_filename="random_forest_model"):
    
    if dataset is None:
        print("Error in train_classic: dataset field is None")
        return
    
    dataset_filename = dataset[0] + ".npz"

    if dataset_filename not in os.listdir("./data/interim/"):
        print(f"Error in train_classic: dataset file {dataset_filename} not found.")
        return
    
    X, y = load_features("./data/interim/" + dataset_filename)
    
    if cross_validation is not None: # cross_validation
        if model_name[0] == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            results, fold_metrics = cross_validate_model(X, y, model, n_splits=cross_validation[0], random_seed=42)
            generate_cross_validation_report(results=results, model_name="Random Forest", fold_metrics=fold_metrics)
        #elif model_name == "xgboost":
        else:
            print(f"Model {model_name[0]} not supported. Try random_forest or xgboost instead.")

    #else:
        #dump(model, "./models/" + model_filename + ".joblib")
    
    return


    

if __name__ == "__main__":
    X, y = load_features("data/interim/features_train.npz")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    options = [1, 2]

    while(1):
        option = input("Which model do you want to train:\n\t1. Random Forest Classifier\n\tXGBoost\n\n\tOption: ")
        if option not in options:
            print("Wrong option!")
        else:
            break
     

    if option == 1:
        model = train_random_forest(X_train, y_train)
    elif option == 2:
        model = train_xgboost(X_train, y_train)

    y_pred = model.predict(X_val)

    dump(model, "models/classic_model.joblib")
