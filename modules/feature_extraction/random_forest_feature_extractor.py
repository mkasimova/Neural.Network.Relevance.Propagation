import logging
import sys

from sklearn.ensemble import RandomForestClassifier

from modules.feature_extraction.feature_extractor import FeatureExtractor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("RF featurizer")


class RandomForestFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, filter_by_distance_cutoff=True, filter_by_DKL=True, filter_by_KS_test=True, n_estimators=30, njobs=4, name="RF"):
        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, filter_by_DKL=filter_by_DKL, filter_by_KS_test=filter_by_KS_test, name=name)
        self.n_estimators = n_estimators
        self.njobs = njobs
        return

    def train(self, train_set, train_labels):
        # Construct and train classifier
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.njobs)
        classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        return classifier.feature_importances_
