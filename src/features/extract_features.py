import numpy as np
import cv2

from scipy.stats import skew, kurtosis

from skimage.measure import shannon_entropy
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

def statisticalFeatures(image: list):
	r"""Extracts statistical features from an image.

    Parameters
    ----------
	image : array_like of real numbers.
		This represents an image with three spectral bands (usually RGB).

    Returns
    -------
    features : list of floats
		A list containing statistical features for each spectral band.

    Notes
    -----
	For each band (R, G, B), it includes:
        
           - mean
           - median
           - standard deviation
           - variance
           - sum
           - skewness
           - kurtosis
           - Shannon entropy
           - 10th percentile
           - 25th percentile
           - 75th percentile
           - 90th percentile
           - interquartile range (75th - 25th)
           - minimum
           - maximum
           - range (max - min)

    The final list has 3 x 16 = 48 features.
	"""

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

def textureFeatures(image: list, levels=256, distances=[1,2,3], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):

	features = []

	image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image_gray = ((image_gray - image_gray.min()) / (image_gray.max() - image_gray.min()) * (levels - 1)).astype(np.uint16)	# Normalize to 0 - (levels - 1)

	for distance in distances:
		for angle in angles:

			glcm = graycomatrix(image_gray, distances=[distance], angels=[angle], levels=levels, symmetric=True, normed=True)

			features += [
				graycoprops(glcm, "contrast")[0, 0],
				graycoprops(glcm, "dissimilarity")[0, 0],
				graycoprops(glcm, "homogeneity")[0, 0],
				graycoprops(glcm, "ASM")[0, 0],
				graycoprops(glcm, "energy")[0, 0],
				graycoprops(glcm, "correlation")[0, 0]
			]

	return features

def lbpFeatures(image, radius=3, points=8):

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

def gaborFeatures(image, frequencies=[0.1, 0.3, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
	features = []

	for channel in range(3):
		channel_data = image[:, :, channel]

		for frequency in frequencies:
			for theta in orientations:

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

def colorSpaceFeatures(image):

	features = []

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

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

def extractFeatures(image):
	r"""Extracts relevant features features from an image.

	Parameters
	----------
	image : 

	Returns
	-------
	features : 
	"""
	return [
		statisticalFeatures(image),
		textureFeatures(image),
		lbpFeatures(image),
		gaborFeatures(image),
		colorSpaceFeatures(image),
	]