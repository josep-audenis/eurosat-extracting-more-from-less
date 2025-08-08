import numpy as np

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