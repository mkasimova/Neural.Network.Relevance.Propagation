from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from .. import utils as utils, filtering
from sklearn.model_selection import KFold

logger = logging.getLogger("Extracting features")


class FeatureExtractor(object):

    def __init__(self,
                 samples=None,
                 cluster_indices=None,
                 scaling=True,
                 filter_by_distance_cutoff=True,
                 lower_bound_distance_cutoff=filtering.lower_bound_distance_cutoff_default,
                 upper_bound_distance_cutoff=filtering.upper_bound_distance_cutoff_default,
                 use_inverse_distances=True,
                 n_splits=10,
                 n_iterations=10,
                 name='FeatureExtractor',
                 error_limit=5,
                 supervised=True,
                 remove_outliers=False,
                 shuffle_datasets=False):
        if samples is None:
            raise Exception("Samples cannot be None")
        self.samples = samples
        self.cluster_indices = cluster_indices
        self.n_clusters = len(list(set(self.cluster_indices)))
        if len(self.cluster_indices.shape) == 1:
            self.labels = utils.create_class_labels(self.cluster_indices)
        elif len(self.cluster_indices.shape) == 2:
            self.labels = np.copy(self.cluster_indices)
            self.cluster_indices = self.labels.argmax(axis=1)
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.scaling = scaling
        self.filter_by_distance_cutoff = filter_by_distance_cutoff
        self.name = name
        self.error_limit = error_limit
        self.use_inverse_distances = use_inverse_distances
        self.lower_bound_distance_cutoff = lower_bound_distance_cutoff
        self.upper_bound_distance_cutoff = upper_bound_distance_cutoff
        self.remove_outliers = remove_outliers
        self.supervised = supervised
        self.shuffle_datasets = shuffle_datasets
        logger.debug("Initializing superclass FeatureExctractor '%s' with the following parameters: "
                     " n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, lower_bound_distance_cutoff %s, "
                     " upper_bound_distance_cutoff %s, remove_outliers %s, use_inverse_distances %s", shuffle_datasets,
                     name, n_splits, n_iterations, scaling, filter_by_distance_cutoff, lower_bound_distance_cutoff,
                     upper_bound_distance_cutoff, remove_outliers, use_inverse_distances, shuffle_datasets)

    def split_train_test(self):
        """
        Split the data into n_splits training and test sets
        """
        if self.n_splits < 2:
            logger.info("Using all data in training and validation sets")
            all_indices = np.empty((1, len(self.samples)))
            for i in range(len(self.samples)):
                all_indices[0, i] = i
            all_indices = all_indices.astype(int)
            return all_indices, all_indices

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle_datasets)

        train_inds = []
        test_inds = []

        for train_ind, test_ind in kf.split(self.samples):
            train_inds.append(train_ind)
            test_inds.append(test_ind)
        return train_inds, test_inds

    def get_train_test_set(self, train_ind, test_ind):
        """
        Get the train and test set given their sample/label indices
        """
        train_set = self.samples[train_ind, :]
        test_set = self.samples[test_ind, :]

        test_labels = self.labels[test_ind, :]
        train_labels = self.labels[train_ind, :]

        return train_set, test_set, train_labels, test_labels

    def train(self, train_set, train_labels):
        pass

    def get_feature_importance(self, model, samples, labels):
        pass

    def extract_features(self):

        logger.info("Performing feature extraction with %s on data of shape %s", self.name, self.samples.shape)

        # Create a list of feature indices
        # This is needed when filtering is applied and re-mapping is further used
        n_features = self.samples.shape[1]
        original_samples = np.copy(self.samples)

        if self.filter_by_distance_cutoff:
            self.samples, indices_for_filtering = filtering.filter_by_distance_cutoff(self.samples,
                                                                                      lower_bound_cutoff=self.lower_bound_distance_cutoff,
                                                                                      upper_bound_cutoff=self.upper_bound_distance_cutoff,
                                                                                      inverse_distances=self.use_inverse_distances)

        if self.scaling:
            # Note that we must use the same scalers for all data
            # It is important for some methods (relevance propagation in NN) that all data is scaled between 0 and 1
            self.samples = utils.scale(self.samples, remove_outliers=self.remove_outliers)

        train_inds, test_inds = self.split_train_test()
        errors = np.zeros(self.n_splits * self.n_iterations)

        feats = []

        for i_split in range(self.n_splits):

            for i_iter in range(self.n_iterations):

                train_set, test_set, train_labels, test_labels = self.get_train_test_set(train_inds[i_split],
                                                                                         test_inds[i_split])

                # Train model
                model = self.train(train_set, train_labels)

                if self.supervised and hasattr(model, "predict"):
                    # Test classifier
                    error = utils.check_for_overfit(test_set, test_labels, model)
                    errors[i_split * self.n_iterations + i_iter] = error

                    do_compute_importance = errors[i_split * self.n_iterations + i_iter] < self.error_limit

                else:
                    do_compute_importance = True

                if do_compute_importance:
                    # Get features importance
                    feature_importance = self.get_feature_importance(model, train_set, train_labels)
                    feats.append(feature_importance)
                else:
                    logger.warn("At iteration %s of %s error %s is too high - not computing feature importance",
                                i_split * self.n_iterations + i_iter + 1, self.n_splits * self.n_iterations, error)

        feats = np.asarray(feats)

        std_feats = np.std(feats, axis=0)
        feats = np.mean(feats, axis=0)

        if len(feats.shape) == 1 and len(std_feats.shape) == 1:
            feats = feats.reshape((feats.shape[0], 1))
            std_feats = std_feats.reshape((std_feats.shape[0], 1))

        if self.filter_by_distance_cutoff:
            # Remapping features if filtering was applied
            # If no filtering was applied, return feats and std_feats
            feats, std_feats = filtering.remap_after_filtering(feats, std_feats, n_features, indices_for_filtering)

        logger.info("Done with %s", self.name)
        logger.info("------------------------------")
        self.samples = np.copy(original_samples)

        self.feature_importance = feats
        self.std_feature_importance = std_feats
        self.test_set_errors = errors

        return self
