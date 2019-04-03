import logging
import sys

from sklearn.decomposition import PCA

from .feature_extractor import FeatureExtractor
from .. import utils
from ..postprocessing import PostProcessor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("PCA")


class PCAFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 variance_cutoff='auto',
                 n_components=None,
                 name="PCA",
                 **kwargs):
        kwargs['n_iterations'] = 1
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=False,
                                  **kwargs)

        logger.debug("Initializing PCA with the following parameters: n_components %s, variance_cutoff %s",
                     n_components, variance_cutoff)
        self.n_components = n_components
        self.variance_cutoff = variance_cutoff

    def train(self, train_set, train_labels):
        logger.debug("Training PCA with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        model = PCA(n_components=self.n_components)
        model.fit(train_set)
        return model

    def get_feature_importance(self, model, samples, labels):
        logger.debug("Extracting feature importance using PCA ...")
        importance = utils.compute_feature_importance_from_components(model.explained_variance_ratio_,
                                                                      model.components_,
                                                                      self.variance_cutoff)
        return importance

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
                             supervised=False)
