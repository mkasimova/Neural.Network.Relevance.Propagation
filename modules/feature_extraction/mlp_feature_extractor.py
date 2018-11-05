from __future__ import absolute_import, division, print_function

import numpy as np
import sklearn

import modules.relevance_propagation as relprop
from modules.feature_extraction.feature_extractor import FeatureExtractor


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, filter_by_distance_cutoff=True, filter_by_DKL=True, filter_by_KS_test=True, hidden_layer_sizes=(100,), randomize=True, name="MLP"):
        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, filter_by_DKL=filter_by_DKL, filter_by_KS_test=filter_by_KS_test, name=name)
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
        # Calculate relevance
        relevance = relprop.relevance_propagation(weights, \
                                                  biases, \
                                                  data,
                                                  labels)
        # average relevance per cluster
        nclusters = labels.shape[1]
        nfeatures = relevance.shape[1]
        result = np.zeros((nfeatures, nclusters))
        frames_per_cluster = np.zeros((nclusters))

        # Rescale relevance according to min and max relevance in each frame
        for i in range(relevance.shape[0]):
            relevance[i,:] = (relevance[i,:]-np.min(relevance[i,:]))/(np.max(relevance[i,:])-np.min(relevance[i,:])+0.000000001)

        for frame_idx, frame in enumerate(labels):
            cluster_idx = labels[frame_idx].argmax()
            frames_per_cluster[cluster_idx] += 1

        for frame_idx, rel in enumerate(relevance):
            cluster_idx = labels[frame_idx].argmax()
            result[:, cluster_idx] += rel / frames_per_cluster[cluster_idx]
        return result
