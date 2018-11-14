import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import sklearn

import modules.relevance_propagation as relprop
from modules.feature_extraction.feature_extractor import FeatureExtractor


logger = logging.getLogger("rbm")


class RbmFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, randomize=True, n_components=None, filter_by_distance_cutoff=True, contact_cutoff=0.5, use_inverse_distances=True, filter_by_DKL=False, filter_by_KS_test=False, name="RBM"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, use_inverse_distances=use_inverse_distances, filter_by_DKL=filter_by_DKL, filter_by_KS_test=filter_by_KS_test, name=name)
        self.randomize = randomize
        self.n_components = n_components

    def train(self, train_set, train_labels):
        if self.n_components is None:
            self.n_components = int(train_set.shape[1]/10)
            logger.info("Automatically set n_components to %s", self.n_components)
        classifier = sklearn.neural_network.BernoulliRBM(
            random_state=(None if self.randomize else 89274),
            n_components=self.n_components
        )
        classifier.fit(train_set)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        #TODO we should look into the effect of using sigmoid instead of ReLu activation here!
        nframes, nfeatures = data.shape
        weights = [
            1, #np.eye(nfeatures), # Don't use eye for large matrices since it will blow up memory
            classifier.components_.T
        ]
        biases = [
            classifier.intercept_visible_, 
            classifier.intercept_hidden_
        ]
        data_propagation = np.copy(data)
        labels_propagation = classifier.transform(data_propagation)
        # Calculate relevance
        relevance = relprop.relevance_propagation(weights, \
                                                  biases, \
                                                  data_propagation, \
                                                  labels_propagation, use_sigmoid_activation=True)

        # Average relevance per cluster
        result = np.zeros((nfeatures, 1))
        for frame_idx, rel in enumerate(relevance):
            ind_negative = np.where(rel<0)[0]
            rel[ind_negative] = 0
            rel -= rel.min()
            scale = rel.max() - rel.min()
            if scale < 1e-9:
               scale = 1e-9
            rel /= scale
            result[:, 0] += rel / nframes 
        return result
