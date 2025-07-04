import os
import numpy as np
import cv2
import sys

from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

def extract_statistical_features(image):

    features = []

    for channel in range(3):
        channel_data = image[:, :, channel]

        features += [
            np.mean(channel_data),
            np.median(channel_data),
            np.std(channel_data),
            np.var(channel_data),
            np.sum(channel_data),
            
            skew(channel_data.flatten()),
            kurtosis(channel_data.flatten()),
            shannon_entropy(channel_data),
            
            np.percentile(channel_data, 10),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.percentile(channel_data, 90),
            np.percentile(channel_data, 75) - np.percentile(channel_data, 25),

            np.min(channel_data),
            np.max(channel_data),
            np.max(channel_data) - np.min(channel_data)
        ]

    return features

def extract_texture_glcm_features(image):

    features = []

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 32 

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_quantized = ((gray.astype(np.float64) / 255.0) * (levels - 1)).astype(np.uint8)

    for distance in distances:
        for angle in angles:

            glcm = graycomatrix(gray_quantized, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)

            features += [
                graycoprops(glcm, "contrast")[0, 0],
                graycoprops(glcm, "dissimilarity")[0, 0],
                graycoprops(glcm, "homogeneity")[0, 0],
                graycoprops(glcm, "ASM")[0, 0],
                graycoprops(glcm, "energy")[0, 0],
                graycoprops(glcm, "correlation")[0, 0]
            ]

    return features


def extract_texture_measure_features(image):

    features = []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features += [
        np.var(gray.astype(np.float64)),
        np.std(gray.astype(np.float64))
    ]   

    kernel = np.ones((3, 3))
    local_mean = cv2.filter2D(gray.astype(np.float64), - 1, kernel/9)
    local_std = np.sqrt(cv2.filter2D((gray.astype(np.float64) - local_mean)**2, -1, kernel/9))
    
    features += [
        np.mean(local_std),
        np.std(local_std),
        np.max(local_std),
        np.min(local_std)
    ]

    features += [
        np.max(gray) - np.min(gray),
        shannon_entropy(gray)
    ]

    return features


def extract_lbp_features(image, radius=3, points=8):

    features = []

    n_points = points * radius

    for channel in range(3):

        ch_data = image[:, :, channel]
        lbp = local_binary_pattern(ch_data, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))

        features.extend(hist)
        features += [
            np.mean(lbp),
            np.std(lbp),
            np.var(lbp)
        ]

    return features


def extract_gabor_filter_features(image, frequencies=[0.1, 0.3, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []

    for channel in range(3):
        channel_data = image[:, :, channel]
        for frequency in frequencies:
            for theta in orientations:

                # Garbor filter
                real, _ = gabor(channel_data, frequency=frequency, theta=theta)

                features += [
                    np.mean(real),
                    np.std(real),
                    np.var(real),
                    np.max(real),
                    np.min(real),
                    shannon_entropy(np.abs(real))
                ]

    return features


def extract_color_space_features(image):

    features = []

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    for channel in range(3):
        hsv_channel_data = hsv[:, :, channel]
        lab_channel_data = lab[:, :, channel]

        features += [
            np.mean(hsv_channel_data),
            np.mean(lab_channel_data),
            np.std(hsv_channel_data),
            np.std(lab_channel_data),
            np.var(hsv_channel_data),
            np.var(lab_channel_data),
            skew(hsv_channel_data.flatten()),
            skew(lab_channel_data.flatten()),
            kurtosis(hsv_channel_data.flatten()),
            kurtosis(lab_channel_data.flatten())
        ]

    return features


def extract_spectral_features(image):
    features = []

    r_band = image[:, :, 0]
    g_band = image[:, :, 1]
    b_band = image[:, :, 2]

    epsilon = 1e-8  # Avoid division by 0

    # Band ratios
    features += [
        np.mean(r_band / (g_band + epsilon)),
        np.mean(g_band / (b_band + epsilon)),
        np.mean(r_band / (b_band + epsilon)),
        np.mean((g_band - r_band) / (g_band + r_band + epsilon)),
        np.mean((r_band + g_band + b_band) / 3) 
    ]

    features += [
        np.std(r_band / (g_band + epsilon)),
        np.std(g_band / (b_band + epsilon)),
        np.std(r_band / (b_band + epsilon))
    ]

    return features


def extract_edge_features(image):
    features = []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel edge
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    features += [
        np.mean(sobel_mag),
        np.std(sobel_mag),
        np.var(sobel_mag),
        np.sum(sobel_mag > np.percentile(sobel_mag, 90))    # Strong edges count
    ]

    # Canny edges
    canny = cv2.Canny(gray.astype(np.uint8), 50, 150)

    features += [
        np.sum(canny > 0) / canny.size, # Edge density
        np.mean(canny),
        np.std(canny),
        np.var(canny)
    ]

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    features += [
        np.mean(np.abs(laplacian)),
        np.std(laplacian),
        np.var(laplacian)
    ]

    return features


def extract_morphological_features(image):
    features = []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)   # Morphological operations
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    features += [
        np.sum(opening > 0) / opening.size,
        np.sum(closing > 0) / closing.size,
        np.mean(opening),
        np.mean(closing),
        np.std(opening),
        np.std(closing)
    ]

    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    features += [
        np.sum(gradient > 0) / gradient.size,
        np.mean(gradient),
        np.std(gradient)
    ]

    return features


def extract_frequency_features(image):
    features = []

    for channel in range(3):

        channel_data = image[:, :, channel]
        fft = np.fft.fft2(channel_data) # Fast Fourier Transform
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Frequency domain statistics
        features += [
            np.mean(magnitude),
            np.std(magnitude),
            np.var(magnitude),
            np.max(magnitude),
            np.sum(magnitude > np.percentile(magnitude, 95))
        ]

        psd = magnitude**2  # Power spectral density

        features += [
            np.mean(psd),
            np.std(psd),
            np.sum(psd) / psd.size
        ]

    return features



def extract_features(path, statistical=True, texture_glcm=True, texture_measure=True, lbp=True, gabor=True, color_space=True, spectral_features=True, edge_features=True, morphological=True):
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
    
    return features 


def generate_features_dataset(output_filename="features_train"):

    X = []
    y = []

    affirmative = ['y', '']

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
        "morphological": input("Use morphological features? (y (default)/n): ").lower() in affirmative
    }

    dataset_dir = "./data/external/EuroSAT/"

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
                               features_config["morphological"])
            X.append(features)
            y.append(category)
            percent = 100 * ((i+1) / images_size)
            filled_length = int(20 * (i+1) // images_size)
            bar = '█' * filled_length + '-' * (20 - filled_length)
            sys.stdout.write(f"\r{category.capitalize()} |{bar}| {percent:.2f}%")
            sys.stdout.flush()
        print()

    np.savez(output_file, X=X, y=y)


if __name__ == "__main__":
    generate_features_dataset("features_train")
