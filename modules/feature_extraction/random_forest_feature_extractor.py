import logging
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from .feature_extractor import FeatureExtractor
from ..postprocessing import PostProcessor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("RF")


class RandomForestFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, filter_by_distance_cutoff=False,
                 contact_cutoff=0.5, n_estimators=30, njobs=-1, randomize=True, name="RF", one_vs_rest=True, remove_outliers=False):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling,
                                  filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name=name,
                                  supervised=True,
                                  remove_outliers=remove_outliers)

        logger.debug("Initializing RF with the following parameters: \
                      n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      n_estimators %s, njobs %s, remove_outliers %s", \
                      n_splits, n_iterations, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      n_estimators, njobs, remove_outliers)

        self.n_estimators = n_estimators
        self.njobs = njobs
        self.randomize=randomize
        self.one_vs_rest = one_vs_rest
        return

    def _train_one_vs_rest(self,data, labels):
        n_clusters = labels.shape[1]
        n_points = data.shape[0]

        classifiers = []

        for i_cluster in range(n_clusters):
            classifiers.append(RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.njobs,
                                              random_state=(None if self.randomize else 89274)))
            tmp_labels = np.zeros(n_points)
            tmp_labels[labels[:, i_cluster] == 1] = 1

            classifiers[i_cluster].fit(data, tmp_labels)

        return classifiers

    def train(self, train_set, train_labels):
        # Construct and train classifier
        logger.debug("Training RF with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        if self.one_vs_rest:
            return self._train_one_vs_rest(train_set, train_labels)
        else:
            classifier = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.njobs,
                                            random_state=(None if self.randomize else 89274))
            classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using RF ...")
        if self.one_vs_rest:
            n_clusters = len(classifier)
            n_features = data.shape[1]
            feature_importances = np.zeros((n_features,n_clusters))
            for i_cluster in range(n_clusters):
                feature_importances[:,i_cluster] = classifier[i_cluster].feature_importances_
            return feature_importances
        else:
            return classifier.feature_importances_

    def postprocessing(self, working_dir=None, rescale_results=True, filter_results=False, feature_to_resids=None, pdb_file=None, predefined_relevant_residues=None, use_GMM_estimator=True, supervised=True):

        return PostProcessor(extractor=self, \
                             working_dir=working_dir, \
                             rescale_results=rescale_results, \
                             filter_results=filter_results, \
                             feature_to_resids=feature_to_resids, \
                             pdb_file=pdb_file, \
                             predefined_relevant_residues=predefined_relevant_residues, \
                             use_GMM_estimator=use_GMM_estimator, \
                             supervised=True)

