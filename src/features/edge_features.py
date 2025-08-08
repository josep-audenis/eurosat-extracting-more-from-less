import cv2
import numpy as np

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