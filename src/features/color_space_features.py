import cv2
import numpy as np

from scipy.stats import skew, kurtosis

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