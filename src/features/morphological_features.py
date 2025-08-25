import cv2

import numpy as np

def extract_morphological_features(image):
    features = []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    for k in [3, 5, 7]:
        kernel = np.ones((k, k), np.uint8)   # Morphological operations
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