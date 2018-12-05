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

import modules.relevance_propagation as relprop
from modules.feature_extraction.feature_extractor import FeatureExtractor

logger = logging.getLogger("mlp")


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True,
                 filter_by_distance_cutoff=True,
                 contact_cutoff=None,
                 use_inverse_distances=True,
                 filter_by_DKL=False,
                 filter_by_KS_test=False,
                 name="MLP",
                 hidden_layer_sizes=(100,),
                 solver='lbfgs',
                 activation=relprop.relu,
                 randomize=True,
                 training_max_iter=100000):
        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                  scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                  contact_cutoff=contact_cutoff, use_inverse_distances=use_inverse_distances,
                                  name=name)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.randomize = randomize
        self.solver = solver
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            logger.warn("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.training_max_iter = training_max_iter

    def train(self, train_set, train_labels):
        classifier = sklearn.neural_network.MLPClassifier(
            solver=self.solver,
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=(None if self.randomize else 89274),
            activation=self.activation,
            max_iter=self.training_max_iter)

        classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        layers = _create_layers(self.activation, classifier)
        # Calculate relevance
        propagator = relprop.RelevancePropagator(layers, activation_function=relprop.relu)
        relevance = propagator.propagate(data, labels)
        # average relevance per cluster
        nclusters = labels.shape[1]
        nfeatures = relevance.shape[1]
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


def _create_layers(activation_function, classifier):
    weights = classifier.coefs_
    biases = classifier.intercepts_
    layers = []
    for idx, weight in enumerate(weights):

        if idx == 0:
            l = relprop.FirstLinear(weight, biases[idx])
        elif activation_function == relprop.relu:
            l = relprop.NextLinear(weight, biases[idx])
        elif activation_function == relprop.logistic_sigmoid:
            l = relprop.FirstLinear(weight, biases[idx])
        else:
            raise Exception(
                "Unsupported activation function {}. Supported values are {}".format(activation_function,
                                                                                     relprop.activation_functions))
        layers.append(l)
        if idx < len(weights) - 1:
            # Add activation to all except output layer
            if activation_function == relprop.logistic_sigmoid:
                layers.append(relprop.LogisticSigmoid())
            elif activation_function == relprop.relu:
                layers.append(relprop.ReLU())

    return layers
