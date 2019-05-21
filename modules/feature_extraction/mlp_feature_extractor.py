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
from .. import filtering

logger = logging.getLogger("mlp")


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 name="MLP",
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
        self.backend = "scikit-learn"  # Only available option for now, more to come probably
        logger.debug("Initializing MLP with the following parameters:"
                     " activation function %s, randomize %s, classifier_kwargs %s,"
                     " per_frame_importance_outfile %s, backend %s",
                     activation, randomize, classifier_kwargs, per_frame_importance_outfile, self.backend)
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            Exception("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.randomize = randomize
        self._classifier_kwargs = classifier_kwargs.copy()
        if classifier_kwargs.get('activation', None) is not None and \
                classifier_kwargs.get('activation') != self.activation:
            logger.warn("Conflicting activation properiies. '%s' will be overwritten with '%s'",
                        classifier_kwargs.get('activation'),
                        self.activation)
        self._classifier_kwargs['activation'] = self.activation
        if not self.randomize:
            self._classifier_kwargs['random_state'] = 89274
        self.per_frame_importance_outfile = per_frame_importance_outfile
        self.frame_importances = None

    def train(self, train_set, train_labels):
        logger.debug("Training %s with %s samples and %s features ...", self.name, train_set.shape[0],
                     train_set.shape[1])
        classifier = sklearn.neural_network.MLPClassifier(**self._classifier_kwargs)
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

        if self.per_frame_importance_outfile is not None:
            if self.n_splits != 1:
                logger.error("Cannot average frame importance to outfile if n_splits != 1. n_splits is now set to %s",
                             self.n_splits)
            if self.shuffle_datasets:
                logger.error("Data set has been shuffled, per frame importance will not be properly mapped")
            else:
                # for every feature in every frame...
                self.frame_importances = np.zeros(relevance.shape)

        for frame_idx, rel in enumerate(relevance):
            cluster_idx = labels[frame_idx].argmax()
            result[:, cluster_idx] += rel / frames_per_cluster[cluster_idx]
            if self.frame_importances is not None:
                self.frame_importances[frame_idx] += rel / self.n_iterations
        return result

    def _create_layers(self, classifier):
        weights = classifier.coefs_
        biases = classifier.intercepts_
        layers = []
        for idx, weight in enumerate(weights):

            if idx == 0:
                l = relprop.FirstLinear(min_val=0, max_val=1, weight=weight, bias=biases[idx])
            else:
                l = relprop.layer_for_string(self.activation, weight=weight, bias=biases[idx])
            if l is None:
                raise Exception(
                    "Cannot create layer at index {} for activation function {}".format(idx, self.activation))
            layers.append(l)
            if idx < len(weights) - 1:
                # Add activation to all except output layer
                activation = relprop.layer_activation_for_string(self.activation)
                if activation is None:
                    raise Exception("Unknown activation function {}".format(self.activation))
                layers.append(activation)
            else:
                if self.backend == 'scikit-learn':
                    # For scikit implementation see  # https://stats.stackexchange.com/questions/243588/how-to-apply-softmax-as-activation-function-in-multi-layer-perceptron-in-scikit
                    # or https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
                    out_activation = relprop.layer_activation_for_string(classifier.out_activation_)
                    if out_activation is None:
                        raise Exception("Unknown activation function {}".format(self.activation))
                    layers.append(out_activation)
                else:
                    raise Exception("Unsupported MLP backend {}".format(self.backend))

        self.layers = layers

    def _on_all_features_extracted(self, feats, errors, n_features):
        FeatureExtractor._on_all_features_extracted(self, feats, errors, n_features)

        if self.filter_by_distance_cutoff and self.frame_importances is not None:
            # Note Transpose so we pretend that one frame is one cluster
            imps, _ = filtering.remap_after_filtering(self.frame_importances.T,
                                                      None,
                                                      n_features,
                                                      self.indices_for_filtering)
            self.frame_importances = imps.T

    def postprocessing(self, **kwargs):
        return PerFrameImportancePostProcessor(extractor=self,
                                               per_frame_importance_outfile=self.per_frame_importance_outfile,
                                               frame_importances=self.frame_importances,
                                               **kwargs)
