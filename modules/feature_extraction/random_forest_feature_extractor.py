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

    def __init__(self,
                 name="RF",
                 classifier_kwargs={
                     'n_estimators': 30,
                     'njobs': -1
                 },
                 randomize=True,
                 one_vs_rest=True,
                 **kwargs):

        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=True,
                                  **kwargs)
        self.randomize = randomize
        self.one_vs_rest = one_vs_rest
        self._classifier_kwargs = classifier_kwargs
        logger.debug("Initializing RF with the following parameters: "
                     " randomize %s, one_vs_rest %s, classifier_kwargs %s",
                     randomize, one_vs_rest, classifier_kwargs)

    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_points = data.shape[0]

        classifiers = []

        for i_cluster in range(n_clusters):
            classifiers.append(RandomForestClassifier(**self.get_classifier_kwargs()))
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
            classifier = RandomForestClassifier(**self.get_classifier_kwargs())
            classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using RF ...")
        if self.one_vs_rest:
            n_clusters = len(classifier)
            n_features = data.shape[1]
            feature_importances = np.zeros((n_features, n_clusters))
            for i_cluster in range(n_clusters):
                feature_importances[:, i_cluster] = classifier[i_cluster].feature_importances_
            return feature_importances
        else:
            return classifier.feature_importances_

    def postprocessing(self, working_dir=None, rescale_results=True, filter_results=False, feature_to_resids=None,
                       pdb_file=None, predefined_relevant_residues=None, use_GMM_estimator=True, supervised=True):

        return PostProcessor(extractor=self, \
                             working_dir=working_dir, \
                             rescale_results=rescale_results, \
                             filter_results=filter_results, \
                             feature_to_resids=feature_to_resids, \
                             pdb_file=pdb_file, \
                             predefined_relevant_residues=predefined_relevant_residues, \
                             use_GMM_estimator=use_GMM_estimator, \
                             supervised=True)

    def get_classifier_kwargs(self):
        classifier_kwargs = self._classifier_kwargs.copy()
        if not self.randomize:
            classifier_kwargs['random_state'] = 89274
        return classifier_kwargs