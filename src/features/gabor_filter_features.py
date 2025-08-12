import numpy as np

from skimage.filters import gabor
from skimage.measure import shannon_entropy

def extract_gabor_filter_features(image, frequencies=[0.1, 0.3, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []
    #orientations=[np.pi/4, np.pi/2, 3*np.pi/4] # Testing if 0 does not bring anything
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