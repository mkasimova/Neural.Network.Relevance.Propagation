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

    def __init__(self, samples, cluster_indices, n_splits=10, scaling=False, filter_by_distance_cutoff=True, filter_by_DKL=False, filter_by_KS_test=False, n_components=None, name="PCA"):
        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=1, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, filter_by_DKL=filter_by_DKL, filter_by_KS_test=filter_by_KS_test, name=name)
        self.n_components = n_components
        return

    def train(self, train_set, train_labels):
        # Construct and train PCA
        model = PCA(n_components=self.n_components)
        model.fit(train_set)
        return model

    def get_n_components(self, model):
        """
        Decide the number of components to keep based on a 75% variance cutoff
        """
        explained_var = model.explained_variance_ratio_
        n_components = 1
        total_var_explained = explained_var[0]
        for i in range(1, explained_var.shape[0]):
            if total_var_explained + explained_var[i] < 0.75:
                total_var_explained += explained_var[i]
                n_components += 1
        logger.info('Selecting %s components', n_components)
        return n_components

    def get_feature_importance(self, model, samples, labels):
        n_components = self.n_components
        if (self.n_components is None) or (self.n_components > 1): #TODO why the second part of if condition?
            n_components = self.get_n_components(model)

        feature_importances = np.sum(np.abs(model.components_[0:n_components]), axis=0) #TODO check if it is correct to take abs of components and not sum
        return feature_importances
