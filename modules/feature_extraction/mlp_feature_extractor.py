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
                 per_frame_importance_samples=None,
                 per_frame_importance_labels=None,
                 classifier_kwargs={},
                 **kwargs):
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=supervised,
                                  **kwargs)
        self.backend = "scikit-learn"  # Only available option for now, more to come probably
        logger.debug("Initializing MLP with the following parameters:"
                     " activation function %s, randomize %s, classifier_kwargs %s,"
                     " per_frame_importance_outfile %s, backend %s, per_frame_importance_samples %s",
                     activation, randomize, classifier_kwargs, per_frame_importance_outfile, self.backend,
                     per_frame_importance_samples)
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            Exception("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.randomize = randomize
        self.classifier_kwargs = classifier_kwargs.copy()
        if classifier_kwargs.get('activation', None) is not None and \
                classifier_kwargs.get('activation') != self.activation:
            logger.warn("Conflicting activation properiies. '%s' will be overwritten with '%s'",
                        classifier_kwargs.get('activation'),
                        self.activation)
        self.classifier_kwargs['activation'] = self.activation
        if not self.randomize:
            self.classifier_kwargs['random_state'] = 89274
        self.frame_importances = None
        self.per_frame_importance_outfile = per_frame_importance_outfile
        self.per_frame_importance_samples = per_frame_importance_samples
        self.per_frame_importance_labels = per_frame_importance_labels

    def train(self, train_set, train_labels):
        logger.debug("Training %s with %s samples and %s features ...", self.name, train_set.shape[0],
                     train_set.shape[1])
        classifier = sklearn.neural_network.MLPClassifier(**self.classifier_kwargs)
        classifier.fit(train_set, train_labels)
        return classifier

    def _normalize_relevance_per_frame(self, relevance_per_frame):
        for i in range(relevance_per_frame.shape[0]):
            ind_negative = np.where(relevance_per_frame[i, :] < 0)[0]
            relevance_per_frame[i, ind_negative] = 0
            relevance_per_frame[i, :] = (relevance_per_frame[i, :] - np.min(relevance_per_frame[i, :])) / \
                                        (np.max(relevance_per_frame[i, :]) - np.min(relevance_per_frame[i, :]) + 1e-9)
        return relevance_per_frame

    def _perform_lrp(self, classifier, data, labels):
        nclusters = labels.shape[1]
        nfeatures = data.shape[1]
        relevance_per_cluster = np.empty((nfeatures, nclusters))
        per_frame_relevance = np.zeros(data.shape)
        for c_idx in range(nclusters):
            # Get all frames belonging to a cluster
            frame_indices = labels[:, c_idx] == 1
            cluster_data = data[frame_indices]
            cluster_labels = np.zeros((len(cluster_data), nclusters))
            cluster_labels[:, c_idx] = 1  # Only look at one class at the time
            # Now see what makes these frames belong to that class
            # Time for LRP
            layers = self._create_layers(classifier)
            propagator = relprop.RelevancePropagator(layers)
            cluster_frame_relevance = propagator.propagate(cluster_data, cluster_labels)
            # Rescale relevance according to min and max relevance in each frame
            cluster_frame_relevance = self._normalize_relevance_per_frame(cluster_frame_relevance)
            relevance_per_cluster[:, c_idx] = cluster_frame_relevance.mean(axis=0)
            per_frame_relevance[frame_indices] += cluster_frame_relevance
        per_frame_relevance = self._normalize_relevance_per_frame(per_frame_relevance)
        return per_frame_relevance, relevance_per_cluster

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using MLP ...")
        relevance_per_frame, relevance_per_cluster = self._perform_lrp(classifier, data, labels)

        if self.per_frame_importance_outfile is not None:
            if self.per_frame_importance_samples is not None:
                other_labels = classifier.predict(self.per_frame_importance_samples) \
                    if self.per_frame_importance_labels is None \
                    else self.per_frame_importance_labels
                other_samples = self.scaler.transform(self.per_frame_importance_samples)
                frame_relevance, _ = self._perform_lrp(classifier, other_samples, other_labels)
            else:
                logger.info("Using same trajectory for per frame importance as was used for training.")
                if self.n_splits != 1:
                    logger.error(
                        "Cannot average frame importance to outfile if n_splits != 1. n_splits is now set to %s",
                        self.n_splits)
                if self.shuffle_datasets:
                    logger.error("Data set has been shuffled, per frame importance will not be properly mapped")
                frame_relevance = relevance_per_frame
            # for every feature in every frame...
            self.frame_importances = np.zeros(relevance_per_frame.shape)
            for frame_idx, rel in enumerate(frame_relevance):
                self.frame_importances[frame_idx] += rel / self.n_iterations
        return relevance_per_cluster

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

        return layers

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
