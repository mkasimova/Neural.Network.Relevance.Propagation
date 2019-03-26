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
from ..postprocessing import PerFrameImportancePostProcessor

logger = logging.getLogger("mlp")


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 name="MLP",
                 hidden_layer_sizes=(100,),
                 activation=relprop.relu,
                 randomize=True,
                 supervised=True,
                 per_frame_importance_outfile=None,
                 classifier_kwargs={},
                 **kwargs):
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=supervised,
                                  **kwargs)
        logger.debug("Initializing MLP with the following parameters:"
                     " hidden_layer_sizes %s, activation function %s, randomize %s, classifier_kwargs %s,"
                     " per_frame_importance_outfile %s",
                     hidden_layer_sizes, activation, randomize, classifier_kwargs, per_frame_importance_outfile)
        self.hidden_layer_sizes = hidden_layer_sizes
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            Exception("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.randomize = randomize
        self.per_frame_importance_outfile = per_frame_importance_outfile
        if self.frame_importances is None:
            if self.n_splits != 1:
                raise Exception("Cannot write frame importance to outfile if n_splits is not 1 ")
            # for every feature in every frame...
            self.frame_importances = np.zeros((self.samples.shape[0], self.samples.shape[1]))
        else:
            self.frame_importances = None
        self._classifier_kwargs = classifier_kwargs

    def train(self, train_set, train_labels):
        logger.debug("Training MLP with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        classifier = sklearn.neural_network.MLPClassifier(**self.get_classifier_kwargs())
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
            if self.frame_importances is not None:
                # Note
                self.frame_importances[frame_idx] += rel / self.n_iterations
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

    def postprocessing(self, working_dir=None, rescale_results=True, filter_results=False, feature_to_resids=None,
                       pdb_file=None, predefined_relevant_residues=None, use_GMM_estimator=True, supervised=True):

        return PerFrameImportancePostProcessor(extractor=self, \
                             working_dir=working_dir, \
                             rescale_results=rescale_results, \
                             filter_results=filter_results, \
                             feature_to_resids=feature_to_resids, \
                             pdb_file=pdb_file, \
                             predefined_relevant_residues=predefined_relevant_residues, \
                             use_GMM_estimator=use_GMM_estimator, \
                             supervised=True,
                             per_frame_importance_outfile=self.per_frame_importance_outfile,
                             frame_importances=self.frame_importances)

    def get_classifier_kwargs(self):
        classifier_kwargs = self._classifier_kwargs.copy()
        classifier_kwargs['activation'] = self.activation
        classifier_kwargs['hidden_layer_sizes'] = self.hidden_layer_sizes
        if not self.randomize:
            classifier_kwargs['random_state'] = 89274
        return classifier_kwargs

