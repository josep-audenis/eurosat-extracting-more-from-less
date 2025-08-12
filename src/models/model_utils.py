import numpy as np

def load_features(feature_file):
    data = np.load(feature_file)
    return data['X'], data['y']