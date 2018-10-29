from __future__ import absolute_import, division, print_function

import numpy as np
import sklearn
import modules.relevance_propagation as relprop
from modules.feature_extraction.feature_extractor import FeatureExtractor


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, labels, n_splits=10, n_iterations=3, scaling=True, hidden_layer_sizes=(100,), randomize=True,
                 name="MLP"):
        FeatureExtractor.__init__(self, samples, labels, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, name=name)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.randomize = randomize


    def train(self, train_set, train_labels):
        classifier = sklearn.neural_network.MLPClassifier(
            solver='lbfgs',
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=(None if self.randomize else 89274),
            activation='relu',
            max_iter=500)

        classifier.fit(train_set, train_labels)
        return classifier


    def get_feature_importance(self, classifier, data, labels):
        weights = classifier.coefs_
        biases = classifier.intercepts_
        # TODO do we need to copy this here?
        data_propagation = np.copy(data)
        labels_propagation = np.copy(labels)
        # Calculate relevance
        relevance = relprop.relevance_propagation(weights, \
                                                                biases, \
                                                                data_propagation,
                                                                labels_propagation)
        #average relevance per cluster
        nclusters = labels.shape[1]
        nfeatures = relevance.shape[1]
        result = np.zeros((nfeatures, nclusters))
        frames_per_cluster = np.zeros((nclusters,))
        for frame_idx, frame in enumerate(labels):
            cluster_idx = labels[frame_idx].argmax()
            frames_per_cluster[cluster_idx] += 1

        for frame_idx, rel in enumerate(relevance):
            cluster_idx = labels[frame_idx].argmax()
            result[:, cluster_idx] = rel/frames_per_cluster[cluster_idx]
        return result
