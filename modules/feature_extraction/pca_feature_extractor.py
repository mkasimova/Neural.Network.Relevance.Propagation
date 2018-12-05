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

logger = logging.getLogger("PCA featurizer")


class PCAFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, scaling=True, filter_by_distance_cutoff=False, contact_cutoff=0.5, n_components=None, name="PCA"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=1, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name=name)
        self.n_components = n_components
        return

    def train(self, train_set, train_labels):
        # Construct and train PCA
        model = PCA(n_components=self.n_components)
        model.fit(train_set)
        return model

    def _get_n_components(self, model, n_clusters):
        """
        Decide the number of components to keep based on a 75% variance cutoff
        """
        explained_var = model.explained_variance_ratio_
        n_components = 1
        total_var_explained = explained_var[0]
        for i in range(1, explained_var.shape[0]):
            if total_var_explained + explained_var[i] < 0.75 and i < n_clusters:
                total_var_explained += explained_var[i]
                n_components += 1
        return n_components

    def _collect_components(self, model, n_components):
        components = np.abs(model.components_[0:n_components] * model.explained_variance_[0:n_components, np.newaxis])
        return components.T

    def get_feature_importance(self, model, samples, labels):
        n_components = self.n_components
        if (self.n_components is None):
            n_clusters = labels.shape[1]
            n_components = self._get_n_components(model, n_clusters)
        logger.info('n_components: '+str(n_components))
        feature_importances = self._collect_components(model, n_components)
        # Removed: summing over eigenvectors:
        #feature_importances = np.sum(np.abs(model.components_[0:n_components]*model.explained_variance_[0:n_components,np.newaxis]), axis=0)
        return feature_importances
