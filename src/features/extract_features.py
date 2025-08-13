import os
import numpy as np
import cv2
import sys

from sklearn.feature_selection import mutual_info_classif

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import CONFIG

from src.features.statistical_features import extract_statistical_features
from src.features.texture_features import extract_texture_glcm_features, extract_texture_measure_features
from src.features.lbp_features import extract_lbp_features
from src.features.gabor_filter_features import extract_gabor_filter_features
from src.features.color_space_features import extract_color_space_features
from src.features.spectral_features import extract_spectral_features
from src.features.edge_features import extract_edge_features
from src.features.morphological_features import extract_morphological_features
from src.features.frequency_features import extract_frequency_features

def extract_features(path, statistical=True, texture_glcm=True, texture_measure=True, lbp=True, gabor=True, color_space=True, spectral_features=True, edge_features=True, morphological=True, frequency=True):
    features = []

    image = cv2.imread(path)

    if statistical:
        features += extract_statistical_features(image)
    if texture_glcm:
        features += extract_texture_glcm_features(image)
    if texture_measure:
        features += extract_texture_measure_features(image)
    if lbp:
        features += extract_lbp_features(image)
    if gabor:
        features += extract_gabor_filter_features(image)
    if color_space:
        features += extract_color_space_features(image)
    if spectral_features:
        features += extract_spectral_features(image)
    if edge_features:
        features += extract_edge_features(image)
    if morphological:
        features += extract_morphological_features(image)
    if frequency:
        features += extract_frequency_features(image)
    return features 


def generate_features_dataset():

    X = []
    y = []

    affirmative = ['y', 'Y', '']

    print("=== Feature selection ===")
    features_config = {
        "statistical": input("Use statistical features? (y (default)/n): ").lower() in affirmative,
        "texture_glcm": input("Use GLCM texture features? (y (default)/n): ").lower() in affirmative,
        "texture_measure": input("Use texture measure features? (y (default)/n): ").lower() in affirmative,
        "lbp": input("Use LBP features? (y (default)/n): ").lower() in affirmative,
        "gabor": input("Use Gabor features? (y (default)/n): ").lower() in affirmative,
        "color_space": input("Use color space features? (y (default)/n): ").lower() in affirmative,
        "spectral_features": input("Use spectral features? (y (default)/n): ").lower() in affirmative,
        "edge_features": input("Use edge features? (y (default)/n): ").lower() in affirmative,
        "morphological": input("Use morphological features? (y (default)/n): ").lower() in affirmative,
        "frequency": input("Use frequency features? (y (default)/n): ").lower() in affirmative
    }

    output_filename = input("\nName of the dataset to generate (default \"features\"): ")

    output_filename = output_filename.replace(" ", "_")

    if output_filename == '':
        output_filename = "features"

    dataset_dir = "./data/raw/EuroSAT/"

    output_file = "./data/interim/" + output_filename + ".npz"

    categories = os.listdir(dataset_dir)
    
    print("\n=== Extraction progress ===\n")

    for category in categories:
        images = os.listdir(dataset_dir + category + "/")
        images_size = len(images)
        for i, image in enumerate(images):
            path = dataset_dir + category + "/" + image
            features = extract_features(path, 
                               features_config["statistical"], 
                               features_config["texture_glcm"], 
                               features_config["texture_measure"],
                               features_config["lbp"],
                               features_config["gabor"],        
                               features_config["color_space"],
                               features_config["spectral_features"],
                               features_config["edge_features"],
                               features_config["morphological"],
                               features_config["frequency"])
            X.append(features)
            y.append(category)
            percent = 100 * ((i+1) / images_size)
            filled_length = int(20 * (i+1) // images_size)
            bar = '█' * filled_length + '-' * (20 - filled_length)
            sys.stdout.write(f"\r{category.capitalize()} |{bar}| {percent:.2f}%")
            sys.stdout.flush()
        print()
    print()
    
    print(f"Total computed features: {len(X[0])}")
    X, mask_nonvariant = remove_nonovariant_features(X)
    if len(np.where(~mask_nonvariant)[0]) != 0:
        print(f"Indices of non variant features:\n{np.where(~mask_nonvariant)[0]}")
    X, mask_mutualinfo = select_features_mutual_info(X,y)
    if len(np.where(~mask_mutualinfo)[0]) != 0:
        print(f"Indices of mutual information features (without non variant features removed previously):\n{np.where(~mask_mutualinfo)[0]}")
    np.savez(output_file, X=X, y=y)
    print(f"Total features after cleaning: {len(X[0])}")

def remove_nonovariant_features(X):
    X = np.array(X)
    variances = np.var(X, axis=0)
    mask = variances > float(CONFIG["remove_variance_threshold"])
    removed = len(mask) - np.sum(mask)
    
    if removed > 0:
        print(f"Removed {removed} constant or near-constant feautres.")
    else:
        print("No constant or near-constant features detected")
    
    return X[:, mask], mask

def select_features_mutual_info(X, y):
    X = np.array(X)
    y = np.array(y)

    mi_scores = mutual_info_classif(X, y, random_state=CONFIG["random_seed"])
    indices_sorted = np.argsort(mi_scores)[::-1]
    selected_indices = indices_sorted[:CONFIG["top_k_features"]]

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected_indices] = True

    return X[:, mask], mask


if __name__ == "__main__":
    generate_features_dataset()