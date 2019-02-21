from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import sklearn.neural_network

from .. import relevance_propagation as relprop
from .feature_extractor import FeatureExtractor

logger = logging.getLogger("mlp")


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True,
                 filter_by_distance_cutoff=False,
                 contact_cutoff=0.5,
                 name="MLP",
                 hidden_layer_sizes=(100,),
                 solver='lbfgs',
                 activation=relprop.relu,
                 randomize=True,
                 training_max_iter=100000,
                 remove_outliers=False):
        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                  scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                  contact_cutoff=contact_cutoff,
                                  name=name,
                                  is_unsupervised=False,
                                  remove_outliers=remove_outliers)
        logger.debug("Initializing MLP with the following parameters: \
                      n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      hidden_layer_sizes %s, solver %s, activation function %s, randomize %s, training_max_iter %s, remove_outliers %s", \
                      n_splits, n_iterations, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      hidden_layer_sizes, solver, activation, randomize, training_max_iter, remove_outliers)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            Exception("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.randomize = randomize
        self.training_max_iter = training_max_iter

    def train(self, train_set, train_labels):
        logger.debug("Training MLP with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        classifier = sklearn.neural_network.MLPClassifier(
            solver=self.solver,
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=(None if self.randomize else 89274),
            activation=self.activation,
            max_iter=self.training_max_iter)

        classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using MLP ...")
        self._create_layers(classifier)
        # Calculate relevance
        propagator = relprop.RelevancePropagator(self.layers)
        relevance = propagator.propagate(data, labels)
        # Average relevance per cluster
        nclusters = labels.shape[1]
        nfeatures = relevance.shape[1]
        result = np.zeros((nfeatures, nclusters))
        frames_per_cluster = np.zeros((nclusters))

        # Rescale relevance according to min and max relevance in each frame
        logger.debug("Rescaling feature importance extracted using MLP in each frame between min and max ...")
        for i in range(relevance.shape[0]):
            ind_negative = np.where(relevance[i, :] < 0)[0]
            relevance[i, ind_negative] = 0
            relevance[i, :] = (relevance[i, :] - np.min(relevance[i, :])) / (
                    np.max(relevance[i, :]) - np.min(relevance[i, :]) + 1e-9)

        for frame_idx, frame in enumerate(labels):
            cluster_idx = labels[frame_idx].argmax()
            frames_per_cluster[cluster_idx] += 1

        for frame_idx, rel in enumerate(relevance):
            cluster_idx = labels[frame_idx].argmax()
            result[:, cluster_idx] += rel / frames_per_cluster[cluster_idx]
        return result

    def _create_layers(self, classifier):
        weights = classifier.coefs_
        biases = classifier.intercepts_
        layers = []
        for idx, weight in enumerate(weights):

            if idx == 0:
                l = relprop.FirstLinear(weight, biases[idx])
            elif self.activation == relprop.relu:
                l = relprop.NextLinear(weight, biases[idx])
            elif self.activation == relprop.logistic_sigmoid:
                l = relprop.FirstLinear(weight, biases[idx])
            layers.append(l)
            if idx < len(weights) - 1:
                # Add activation to all except output layer
                if self.activation == relprop.logistic_sigmoid:
                    layers.append(relprop.LogisticSigmoid())
                elif self.activation == relprop.relu:
                    layers.append(relprop.ReLU())

        self.layers = layers
