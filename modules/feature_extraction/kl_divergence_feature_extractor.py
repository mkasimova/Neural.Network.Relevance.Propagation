from __future__ import absolute_import, division, print_function

import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.stats import entropy
from modules.feature_extraction.feature_extractor import FeatureExtractor

logger = logging.getLogger("KL")

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
    bin_min = min(x.min(), y.min())
    bin_max = max(x.max(), y.max())
    if bin_size >= (bin_max - bin_min):
        DKL = 0
    else:
        bin_n = int((bin_max - bin_min) / bin_size)
        x_prob = np.histogram(x, bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 0.000000001
        y_prob = np.histogram(y, bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 0.000000001
        DKL = 0.5 * (entropy(x_prob, y_prob) + entropy(y_prob, x_prob))
    return DKL


class KLFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, labels, n_splits=10, n_iterations=3, scaling=True, bin_size=None):
        FeatureExtractor.__init__(self, samples, labels, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, name="KL")
        self.bin_size = bin_size

    def train(self, train_set, train_labels):
        pass

    def get_feature_importance(self, classifier, data, labels):
        nclusters = labels.shape[1]
        nfeatures = data.shape[1]
        relevance = np.zeros((nfeatures, nclusters))
        for c1 in range(nclusters):
            data_c1 = data[labels[:,c1]>0]
            for c2 in range(c1 + 1, nclusters):
                data_c2 = data[labels[:,c2]>0]
                #print(data_c1.shape, data_c2.shape)
                if len(data_c1) == 0 or len(data_c2) == 0:
                    logger.warn("Unbalanced data partitioning. No data for one cluster. Ignoring...")
                    continue
                dkl =  KL_divergence(data_c1, data_c2, self.bin_size)
                relevance[:,[c1,c2]] += dkl #add relevance for both these clusters
                
        return relevance
                
