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

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, randomize=True,
                 n_components=None, filter_by_distance_cutoff=True, contact_cutoff=0.5, use_inverse_distances=True,
                 filter_by_DKL=False, filter_by_KS_test=False, name="RBM"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                  scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                  contact_cutoff=contact_cutoff, use_inverse_distances=use_inverse_distances,
                                  filter_by_DKL=filter_by_DKL, filter_by_KS_test=filter_by_KS_test, name=name)
        self.randomize = randomize
        self.n_components = n_components

    def train(self, train_set, train_labels):
        if self.n_components is None:
            self.n_components = int(train_set.shape[1] / 10)
            logger.info("Automatically set n_components to %s", self.n_components)

        classifier = sklearn.neural_network.BernoulliRBM(
            random_state=(None if self.randomize else 89274),
            n_components=self.n_components
        )
        classifier.fit(train_set)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        # TODO we should look into the effect of using sigmoid instead of ReLu activation here!
        nframes, nfeatures = data.shape
        classifier.intercept_hidden_

        labels_propagation = classifier.transform(data) #same as perfect classification

        # Calculate relevance
        # see https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html
        layers = [
            relprop.FirstLinear(classifier.components_.T, classifier.intercept_hidden_),
            relprop.LogisticSigmoid()
        ]
        propagator = relprop.RelevancePropagator(layers, activation_function=relprop.logistic_sigmoid)
        relevance = propagator.propagate(data, labels_propagation)

        # Average relevance per cluster
        nclusters = labels.shape[1]

        result = np.zeros((nfeatures, nclusters))
        frames_per_cluster = np.zeros((nclusters))

        # Rescale relevance according to min and max relevance in each frame
        for i in range(relevance.shape[0]):
            ind_negative = np.where(relevance[i, :] < 0)[0]
            relevance[i, ind_negative] = 0
            relevance[i, :] = (relevance[i, :] - np.min(relevance[i, :])) / (
                    np.max(relevance[i, :]) - np.min(relevance[i, :]) + 0.000000001)

        for frame_idx, frame in enumerate(labels):
            cluster_idx = labels[frame_idx].argmax()
            frames_per_cluster[cluster_idx] += 1

        for frame_idx, rel in enumerate(relevance):
            cluster_idx = labels[frame_idx].argmax()
            result[:, cluster_idx] += rel / frames_per_cluster[cluster_idx]

        return result
