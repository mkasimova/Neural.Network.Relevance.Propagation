import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import sklearn

import modules.relevance_propagation as relprop
from modules.feature_extraction.feature_extractor import FeatureExtractor
from modules.postprocessing import PostProcessor
from sklearn.neural_network import BernoulliRBM
import scipy

logger = logging.getLogger("rbm")


class RbmFeatureExtractor(FeatureExtractor):

    def __init__(self, samples, cluster_indices, n_components, n_splits=10, n_iterations=10, scaling=True, randomize=True,
                 filter_by_distance_cutoff=False, contact_cutoff=0.5, method="from_lrp",
                 name="RBM"):

        FeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                  scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                  contact_cutoff=contact_cutoff,
                                  supervised=False,
                                  name=name)
        logger.debug("Initializing RBM with the following parameters: \
                      n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      n_components %s, randomize %s, method %s", \
                      n_splits, n_iterations, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      n_components, randomize, method)

        self.n_components = n_components
        self.randomize = randomize
        self.method = method

    def train(self, train_set, train_labels):
        logger.debug("Training RBM with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])

        classifier = BernoulliRBM(
            random_state=(None if self.randomize else 89274),
            n_components=self.n_components)
        classifier.fit(train_set)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using RBM ...")
        logger.info("RBM psuedo-loglikelihood: "+str(classifier.score_samples(data).mean()))
		
        if self.method=="from_lrp":
            nframes, nfeatures = data.shape

            labels_propagation = classifier.transform(data) # same as perfect classification

            # Calculate relevance
            # see https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html
            layers = self._create_layers(classifier)

            propagator = relprop.RelevancePropagator(layers)
            relevance = propagator.propagate(data, labels_propagation)

            # Average relevance per cluster
            nclusters = labels.shape[1]

            result = np.zeros((nfeatures, nclusters))
            frames_per_cluster = np.zeros((nclusters))

            # Rescale relevance according to min and max relevance in each frame
            logger.debug("Rescaling feature importance extracted using RBM in each frame between min and max ...")

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

        elif self.method=="from_components":

            # Extract components and compute their variance
            components = classifier.components_
            projection = scipy.special.expit( np.matmul(data, components.T) )
            components_var = projection.var(axis=0)

            # Sort components according to their variance
            ind_components_var_sorted = np.argsort(-components_var)
            components_var_sorted = components_var[ind_components_var_sorted]
            components_sorted = components[ind_components_var_sorted,:]

            # Consider only those components whose summed variance constitutes 75% of total variance
            var = components_var_sorted[0]
            result = np.abs(components_sorted[0])*components_var_sorted[0]
            components_var_sum = components_var.sum()
            nclusters = labels.shape[1]
            for i in range(1,components_var_sorted.shape[0]):
                if var + components_var_sorted[i] < 0.75*components_var_sum and i < nclusters:
                    var += components_var_sorted[i]
                    result = np.vstack(( result,np.abs(components_sorted[i])*components_var_sorted[i] ))
                else:
                    break
            result = result.T

        return result

    def _create_layers(self, classifier):
        return [ relprop.FirstLinear(classifier.components_.T, classifier.intercept_hidden_),
                 relprop.LogisticSigmoid()
               ]

    def postprocessing(self, working_dir=None, rescale_results=True, filter_results=False, feature_to_resids=None, pdb_file=None, predefined_relevant_residues=None, use_GMM_estimator=True, supervised=True):

        return PostProcessor(extractor=self, \
                             working_dir=working_dir, \
                             rescale_results=rescale_results, \
                             filter_results=filter_results, \
                             feature_to_resids=feature_to_resids, \
                             pdb_file=pdb_file, \
                             predefined_relevant_residues=predefined_relevant_residues, \
                             use_GMM_estimator=use_GMM_estimator, \
                             supervised=False)

