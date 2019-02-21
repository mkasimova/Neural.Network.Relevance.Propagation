from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import sklearn.neural_network

from .. import relevance_propagation as relprop
from .mlp_feature_extractor import MlpFeatureExtractor

logger = logging.getLogger("mlp_ae")


class MlpAeFeatureExtractor(MlpFeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True,
                 filter_by_distance_cutoff=False,
                 contact_cutoff=0.5,
                 name="MLP_AE",
                 hidden_layer_sizes=(100,),
                 solver='lbfgs',
                 activation=relprop.relu,
                 randomize=True,
                 training_max_iter=100000,
                 use_reconstruction_for_lrp=False,
                 remove_outliers=False):
        MlpFeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                     scaling=scaling,
                                     filter_by_distance_cutoff=filter_by_distance_cutoff,
                                     contact_cutoff=contact_cutoff,
                                     name=name,
                                     hidden_layer_sizes=hidden_layer_sizes,
                                     solver=solver,
                                     activation=activation,
                                     randomize=randomize,
                                     training_max_iter=training_max_iter,
                                     remove_outliers=remove_outliers)
        self.is_unsupervised = True
        self.use_reconstruction_for_lrp = use_reconstruction_for_lrp

    def train(self, train_set, train_labels):
        logger.debug("Training MLP with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        classifier = sklearn.neural_network.MLPRegressor(
            solver=self.solver,
            hidden_layer_sizes=list(self.hidden_layer_sizes) + [train_set.shape[1]],
            random_state=(None if self.randomize else 89274),
            activation=self.activation,
            max_iter=self.training_max_iter)

        classifier.fit(train_set, train_set)  # note same output as input
        self.classifier = classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using MLP Autoencoder ...")
        target_values = self.classifier.predict(data) if self.use_reconstruction_for_lrp else data
        res = MlpFeatureExtractor.get_feature_importance(self, self.classifier, data, target_values)
        return res.mean(axis=1)
