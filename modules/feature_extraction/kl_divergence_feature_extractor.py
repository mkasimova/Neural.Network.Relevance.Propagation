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
from .feature_extractor import FeatureExtractor
from ..postprocessing import PostProcessor

logger = logging.getLogger("KL divergence")


class KLFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, scaling=True, filter_by_distance_cutoff=False, contact_cutoff=0.5,
                 cluster_split_method="one_vs_rest", bin_width=None, remove_outliers=False):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=1, scaling=scaling,
                                  filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name="KL",
                                  supervised=True,
                                  remove_outliers=remove_outliers)

        logger.debug("Initializing KL with the following parameters: \
                      n_splits %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      bin_width %s, remove_outliers %s", \
                      n_splits, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      bin_width ,remove_outliers)
        self.bin_width = bin_width
        if bin_width is None:
            logger.debug('Using standard deviation of each feature as bin size.')
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

        DKL = np.zeros(n_features)
        if self.bin_width is not None:
            tmp_bin_width = self.bin_width

        for i_feature in range(n_features):
            xy = np.concatenate((x[:, i_feature], y[:, i_feature]))
            bin_min = np.min(xy)
            bin_max = np.max(xy)

            if self.bin_width is None:
                tmp_bin_width = np.std(x[:,i_feature])
                if tmp_bin_width == 0:
                    tmp_bin_width = 0.1 # Set arbitrary bin width if zero

            if tmp_bin_width >= (bin_max - bin_min):
                DKL[i_feature] = 0
            else:
                bin_n = int((bin_max - bin_min) / tmp_bin_width)
                x_prob = np.histogram(x[:,i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                y_prob = np.histogram(y[:,i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                DKL[i_feature] = 0.5 * (entropy(x_prob, y_prob) + entropy(y_prob, x_prob)) # An alternative is to use max
        return DKL

    def get_feature_importance(self, model, data, labels):
        """
        Get the feature importance of KL divergence by comparing each cluster to all other clusters
        """
        logger.debug("Extracting feature importance using KL ...")
        return self.feature_importances
    
    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_features = data.shape[1]

        self.feature_importances = np.zeros((n_features,n_clusters))
        for i_cluster in range(n_clusters):
            data_cluster = data[labels[:, i_cluster] == 1, :]
            data_rest = data[labels[:, i_cluster] == 0, :]
            self.feature_importances[:,i_cluster] = self._KL_divergence(data_cluster, data_rest)
        return self

    def postprocessing(self, working_dir=None, rescale_results=True, filter_results=False, feature_to_resids=None, pdb_file=None, predefined_relevant_residues=None, use_GMM_estimator=True, supervised=True):

        return PostProcessor(extractor=self, \
                             working_dir=working_dir, \
                             rescale_results=rescale_results, \
                             filter_results=filter_results, \
                             feature_to_resids=feature_to_resids, \
                             pdb_file=pdb_file, \
                             predefined_relevant_residues=predefined_relevant_residues, \
                             use_GMM_estimator=use_GMM_estimator, \
                             supervised=True)

