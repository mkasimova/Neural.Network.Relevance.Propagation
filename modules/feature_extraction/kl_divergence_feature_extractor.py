from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.stats import entropy
from modules.feature_extraction.feature_extractor import FeatureExtractor

logger = logging.getLogger("KL divergence")


class KLFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, scaling=True, filter_by_distance_cutoff=False, contact_cutoff=0.5,
                 cluster_split_method="one_vs_rest", bin_size=None):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=1, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name="KL")
        logger.debug("Initializing KL with the following parameters: \
                      n_splits %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      bin_size %s", \
                      n_splits, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      bin_size)
        self.bin_size = bin_size
        self.feature_importances = None
        self.cluster_split_method = cluster_split_method

    def train(self, data, labels):
        logger.debug("Training KL with %s samples and %s features ...", data.shape[0], data.shape[1])
        if self.cluster_split_method == "one_vs_rest":
            self._train_one_vs_rest(data, labels)
        elif self.cluster_split_method == "one_vs_one":
            self._train_one_vs_one(data, labels)
        else:
            raise Exception("Unsupported split method: {}".format(self.cluster_split_method))
      
    def _KL_divergence(self, x, y):
        """
        Compute Kullback-Leibler divergence
        """
        n_features = x.shape[1]
        if self.bin_size is None:
            std = np.zeros(n_features)
            for i_feature in range(n_features):
                std[i_feature] = np.std(x[:, i_feature]) #TODO do per feature
            self.bin_size = np.mean(std)
            logger.debug("bin_size for KL is %s", self.bin_size)

        DKL = np.zeros(n_features)
        for i_feature in range(n_features):
            bin_min = np.min(np.concatenate((x[:, i_feature], y[:, i_feature])))
            bin_max = np.max(np.concatenate((x[:, i_feature], y[:, i_feature])))
            if self.bin_size >= (bin_max - bin_min):
                DKL[i_feature] = 0
            else:
                bin_n = int((bin_max - bin_min) / self.bin_size)
                x_prob = np.histogram(x[:, i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                y_prob = np.histogram(y[:, i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                DKL[i_feature] = 0.5 * (entropy(x_prob, y_prob) + entropy(y_prob, x_prob)) # An alternative is to use max
        return DKL

    def get_feature_importance(self, model, data, labels):
        """
        Get the feature importance of KL divergence by comparing each cluster to all other clusters
        """
        logger.debug("Extracting feature importance using KL ...")
        return self.feature_importances.T
    
    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_features = data.shape[1]

        self.feature_importances = np.zeros((n_clusters, n_features))
        for i_cluster in range(n_clusters):
            data_cluster = data[labels[:, i_cluster] == 1, :]
            data_rest = data[labels[:, i_cluster] == 0, :]
            self.feature_importances[i_cluster, :] = self._KL_divergence(data_cluster, data_rest)
        return self     
