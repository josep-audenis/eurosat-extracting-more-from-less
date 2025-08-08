import numpy as np

from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy


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