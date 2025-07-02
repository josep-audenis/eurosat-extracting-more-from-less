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

def train_classic():
	
	train_files = os.listdir("/data/interim/")

	if (len(train_files) == 0):
		print("There are no feature files created, please extract features with the flag -e/--extract.")
		return
	
	print("Select a features file")
	for i, file in enumerate(train_files):
		print(f"\t{i + 1} - {file}")
	
	selected_file = input("Option: ")
	
	X, y = load_features("/data/interim/" + selected_file)

	model = None
	while model is None:
		option = input("\nWith which model do you want to train?\n\t1. Random Forest\n\t2. XGBoost\nOption: ")
		if option == '1':
			model = train_random_forest()
			
		elif option == '2':
			model = train_xgboost()
		else:
			print("Wrong option!")

	

if __name__ == "__main__":
	X, y = load_features("data/interim/features_train.npz")
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

	rfc = train_random_forest(X_train, y_train)

	y_pred = rfc.predict(X_val)

	dump(rfc, "models/classic_model.joblib")