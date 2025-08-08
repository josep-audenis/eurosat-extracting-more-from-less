import numpy as np

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