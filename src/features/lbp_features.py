import numpy as np

from skimage.feature import local_binary_pattern

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