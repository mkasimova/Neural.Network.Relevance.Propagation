import sys
import logging

import numpy as np
from sklearn.decomposition import PCA

from modules.feature_extraction.feature_extractor import FeatureExtractor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("PCA")


class PCAFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, scaling=True, filter_by_distance_cutoff=False, contact_cutoff=0.5, n_components=None, name="PCA"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=1, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name=name)
        logger.debug("Initializing PCA with the following parameters: \
                      n_splits %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      n_components %s", \
                      n_splits, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      n_components)
        self.n_components = n_components
        return

    def train(self, train_set, train_labels):
        logger.debug("Training PCA with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        model = PCA(n_components=self.n_components)
        model.fit(train_set)
        return model

    def _get_n_components(self, model, n_clusters):
        """
        Decide the number of components to keep based on a 75% variance cutoff and not more than the number of clusters
        """
        explained_var = model.explained_variance_ratio_
        n_components = 1
        total_var_explained = explained_var[0]
        for i in range(1, explained_var.shape[0]):
            if total_var_explained + explained_var[i] < 0.75 and i < n_clusters: #TODO should we keep both conditions?
                total_var_explained += explained_var[i]
                n_components += 1
        return n_components

    def _collect_components(self, model, n_components): #TODO should we make feature extraction separate for PCA and RBM?
        components = np.abs(model.components_[0:n_components] * model.explained_variance_[0:n_components, np.newaxis])
        return components.T

    def get_feature_importance(self, model, samples, labels):
        logger.debug("Extracting feature importance using PCA ...")
        n_components = self.n_components
        if (self.n_components is None):
            n_clusters = labels.shape[1]
            n_components = self._get_n_components(model, n_clusters)
            logger.debug("n_components for PCA is %s", n_components)
        feature_importances = self._collect_components(model, n_components)
        return feature_importances
