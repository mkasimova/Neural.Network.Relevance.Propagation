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

class KLFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, labels, n_splits=10, n_iterations=3, scaling=True, bin_size=None):
        FeatureExtractor.__init__(self, samples, labels, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, name="KL")
        self.bin_size = bin_size

    def train(self, train_set, train_labels):
        pass

    def KL_divergence(self, x, y):
        """
        Compute Kullback-Leibler divergence
        """
        n_features = x.shape[1]
        if self.bin_size is None:
            std = np.zeros(n_features)
            for i_feature in range(n_features):
                std[i_feature] = np.std(x[:, i_feature])
            self.bin_size = np.mean(std)

        DKL = np.zeros(n_features)
        for i_feature in range(n_features):
            bin_min = np.min(np.concatenate((x[:,i_feature], y[:,i_feature])))
            bin_max = np.max(np.concatenate((x[:,i_feature], y[:,i_feature])))
            if self.bin_size >= (bin_max - bin_min):
                DKL[i_feature] = 0
            else:
                bin_n = int((bin_max - bin_min) / self.bin_size)
                x_prob = np.histogram(x[:,i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                y_prob = np.histogram(y[:,i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                DKL[i_feature] = 0.5 * (entropy(x_prob, y_prob) + entropy(y_prob, x_prob))
        return DKL

                
    def get_feature_importance(self, model, data, labels):
        """
        Get the feature importance of KL divergence by comparing each cluster to all other clusters.
        """
        n_clusters = labels.shape[1]

        feature_importances = np.zeros((n_clusters, data.shape[1]))
        for i_cluster in range(n_clusters):
            data_cluster = data[labels[:,i_cluster]==1,:]
            data_rest = data[labels[:,i_cluster]==0,:]
            feature_importances[i_cluster,:] = self.KL_divergence(data_cluster, data_rest)

        return feature_importances.T

    def _alternative_get_feature_importance(self, classifier, data, labels):
        """Alternative method comparing one vs one instead of one vs all"""
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
                
        return relevance.T        
