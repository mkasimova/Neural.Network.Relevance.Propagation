from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.stats import entropy
from modules.feature_extraction.feature_extractor import FeatureExtractor


def KL_divergence(x, y, bin_size):
    """
    Compute Kullback-Leibler divergence
    """
    number_of_features = x.shape[1]
    if bin_size is None:
        std = np.zeros(number_of_features)
        for i in range(number_of_features):
            std[i] = np.std(x[:, i])
        bin_size = np.mean(std)
    bin_min = np.min(np.concatenate((x, y)))
    bin_max = np.max(np.concatenate((x, y)))
    if bin_size >= (bin_max - bin_min):
        DKL = 0
    else:
        bin_n = int((bin_max - bin_min) / bin_size)
        x_prob = np.histogram(x, bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 0.000000001
        y_prob = np.histogram(y, bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 0.000000001
        DKL = 0.5 * (entropy(x_prob, y_prob) + entropy(y_prob, x_prob))
    return DKL


class KLFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, labels, n_splits=10, scaling=True, bin_size=None):
        FeatureExtractor.__init__(self, samples, labels, n_splits=n_splits, scaling=scaling, name="KL")
        self.bin_size = bin_size

    def train(self, train_set, train_labels):
        pass

    def get_feature_importance(self, classifier, data, labels):
        return KL_divergence(data, labels, self.bin_size)
