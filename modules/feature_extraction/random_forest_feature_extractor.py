import logging
import sys

from sklearn.ensemble import RandomForestClassifier

from modules.feature_extraction.feature_extractor import FeatureExtractor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("RF")


class RandomForestFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True, filter_by_distance_cutoff=False, contact_cutoff=0.5, n_estimators=30, njobs=-1, randomize=True, name="RF"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff, contact_cutoff=contact_cutoff, name=name)
        logger.debug("Initializing RF with the following parameters: \
                      n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      n_estimators %s, njobs %s", \
                      n_splits, n_iterations, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      n_estimators, njobs)

        self.n_estimators = n_estimators
        self.njobs = njobs
        self.randomize=randomize
        return

    def train(self, train_set, train_labels):
        # Construct and train classifier
        logger.debug("Training RF with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            n_jobs=self.njobs,
                                            random_state=(None if self.randomize else 89274))
        classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using RF ...")
        return classifier.feature_importances_
