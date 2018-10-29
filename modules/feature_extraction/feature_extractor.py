from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from modules import utils as utils, postprocessing as pp, filtering
from sklearn.model_selection import KFold

logger = logging.getLogger("Extracting feature")


class FeatureExtractor(object):

    def __init__(self, samples, labels=None, scaling=True, n_splits=20, n_iterations=3, name=''):
        # Setting parameters
        self.samples = samples
        self.labels = labels
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.scaling = scaling
        self.name = name

    def split_train_test(self):
        """
        Split the data into n_splits training and test sets
        """
        if self.n_splits < 2:
            logger.info("Using all data in training and validation sets")
            all_indices = np.empty((1, len(self.samples)), )
            for i in range(len(self.samples)):
                all_indices[0, i] = i
            all_indices = all_indices.astype(int)
            return all_indices, all_indices

        kf = KFold(n_splits=self.n_splits, shuffle=False)

        train_inds = []
        test_inds = []

        for train_ind, test_ind in kf.split(self.samples):
            train_inds.append(train_ind)
            test_inds.append(test_ind)
        return train_inds, test_inds

    def get_train_test_set(self, train_ind, test_ind):
        """
        Get the train and test set given their sample/label indices.
        """
        train_set = self.samples[train_ind, :]
        test_set = self.samples[test_ind, :]

        if self.labels is not None:
            test_labels = self.labels[test_ind, :]
            train_labels = self.labels[train_ind, :]
        else:
            test_labels = None
            train_labels = None

        return train_set, test_set, train_labels, test_labels

    def train(self, train_set, train_labels):
        pass

    def get_feature_importance(self, model, samples, labels):
        pass

    def extract_features(self, rescale_results=True, filter_results=True):

        train_inds, test_inds = self.split_train_test()
        errors = np.zeros(self.n_splits * self.n_iterations)

        feats = []
        if self.scaling:
            # Note that we must use the same scalers for all data.
            # It is important for some methods (relevance propagation on NN) that all data is scaled between 0 and 1
            full_set, perc_2, perc_98, scaler = utils.scale(self.samples)
        else:
            full_set = self.samples

        for i_split in range(self.n_splits):
            for i_iter in range(self.n_iterations):

                logger.debug("Iteration %s of %s", i_split * self.n_iterations + i_iter + 1,
                             self.n_splits * self.n_iterations)
                train_set, test_set, train_labels, test_labels = \
                    self.get_train_test_set(train_inds[i_split], test_inds[i_split])
                if self.scaling:
                    train_set, perc_2, perc_98, scaler = utils.scale(train_set, perc_2, perc_98, scaler)
                    test_set, perc_2, perc_98, scaler = utils.scale(test_set, perc_2, perc_98, scaler)

                # Train model
                model = self.train(train_set, train_labels)

                if self.labels is not None and model is not None and hasattr(model, "predict"):
                    # Test classifier
                    error = utils.check_for_overfit(test_set, test_labels, model)
                    errors[i_split * self.n_iterations + i_iter] = error

                    logger.debug("Error: %s", errors[i_split * self.n_iterations + i_iter])
                    do_compute_importance = errors[i_split * self.n_iterations + i_iter] < 5
                else:
                    do_compute_importance = True

                if do_compute_importance:
                    logger.debug("Computing feature importance on all data.")
                    # Get feature importances
                    feature_importance = self.get_feature_importance(model, full_set, self.labels)
                    feats.append(feature_importance)
                else:
                    logger.warn("Error too high - not computing feature importance.");

        feats = np.asarray(feats);
        std_feats = np.std(feats, axis=0)
        feats = np.mean(feats, axis=0);
        if rescale_results:
            feats, std_feats = pp.rescale_feature_importance(feats, std_feats);
        if filter_results:
            feats, std_feats = filtering.filter_feature_importance(feats, std_feats);
        return feats, std_feats, errors
