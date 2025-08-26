import cv2
import numpy as np

from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

def extract_texture_glcm_features(image):

    features = []

    distances = [1, 2, 3, 4]
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