from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np
import networkx as nx
import modules.utils as utils
from scipy.spatial.distance import squareform
from modules import postprocessing
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

logger = logging.getLogger("projection")

class DataProjector():
    def __init__(self, postprocessor, samples):
        """
        Class that performs dimensionality reduction using the relevances from each estimator.
        :param postprocessor:
        :param samples:
        """
        self.pop = postprocessor
        self.samples = samples
        self.labels = self.pop.extractor.labels

        self.n_clusters = self.labels.shape[1]
        self.labels = np.argmax(self.labels,axis=1)

        self.extractor = self.pop.extractor

        self.basis_vector_projection = None
        self.raw_projection = None

        self.separation_score = None
        self.projection_class_entropy = None
        self.cluster_projection_class_entropy = None
        return

    def project(self, do_coloring=False):
        """
        Project distances. Performs:
          1. Raw projection (projection onto cluster feature importances if feature_importances_per_cluster is given)
          2. Basis vector projection (basis vectors identified using graph coloring).
        """

        if self.pop.importance_per_cluster is not None:
            self.raw_projection = self._project_on_relevance_basis_vectors(self.samples, self.pop.importance_per_cluster)

        if do_coloring:
            logger.info("Identifying basis vectors")
            relevance_basis_vectors = self._identify_relevance_basis_vectors(squareform(self.pop.feature_importance[:,0]))
            logger.info("Projecting onto "+str(relevance_basis_vectors.shape[1])+" identified dimensions.")
            if relevance_basis_vectors.shape[1] < 10 and relevance_basis_vectors.shape[0] > 0:
                self.basis_vector_projection = self._project_on_relevance_basis_vectors(self.samples, relevance_basis_vectors)

        return self

    def evaluate_importance_robustness(self):
        """
        Evaluating robustness of importances by removing one feature at a time and computing separation scores and
        total posterior entropy.
        :return:
        """
        # TODO: implement
        return


    def score_projection(self, raw_projection=True, use_GMM=True):
        """
        Score the resulting projection by approximating each cluster as a Gaussian (or Gaussian mixture)
        distribution and classify points using the posterior distribution over classes.
        The number of correctly classified points divided by the number of points is the projection score.
        :return: itself
        """

        if raw_projection:
            proj = np.copy(self.raw_projection)
            logger.info("Scoring raw projections.")
        else:
            proj = np.copy(self.basis_vector_projection)
            logger.info("Scoring basis vector projections.")

        priors = self._set_class_prior()

        if use_GMM:
            GMMs = self._fit_GM(proj)
        else:
            means, covs = self._fit_Gaussians(proj)

        n_points = proj.shape[0]
        new_classes = np.zeros(n_points)
        class_entropies = np.zeros(n_points)
        for i_point in range(n_points):
            if use_GMM:
                posteriors = self._compute_GM_posterior(proj[i_point, :], priors, GMMs)
            else:
                posteriors = self._compute_posterior(proj[i_point,:], priors, means, covs)
            class_entropies[i_point] = entropy(posteriors)
            new_classes[i_point] = np.argmax(posteriors)

        # Compute separation score
        correct_separation = new_classes==self.labels
        self.separation_score = correct_separation.sum()/n_points
        self.projection_class_entropy = class_entropies.mean()

        # Compute per-cluster projection entropy
        self.cluster_projection_class_entropy = np.zeros(self.n_clusters)
        for i_cluster in range(self.n_clusters):
            inds = self.labels==i_cluster
            self.cluster_projection_class_entropy[i_cluster] = class_entropies[inds].mean()

        return

    def persist(self):
        """
        Write projected data to files.
        """
        if self.raw_projection is not None:
            np.save(self.directory + "relevance_raw_projection",self.raw_projection)
        if self.basis_vector_projection is not None:
            np.save(self.directory + "relevance_basis_vector_projection",self.basis_vector_projection)
        return

    def _compute_posterior(self, x, priors, means, covs):
        """
        Compute class posteriors
        :param point:
        :param priors:
        :param means:
        :param covs:
        :return:
        """
        posteriors = np.zeros(self.n_clusters)
        for i_cluster in range(self.n_clusters):
            density = multivariate_normal.pdf(x, mean=means[i_cluster], cov=covs[i_cluster])
            posteriors[i_cluster] = density*priors[i_cluster]

        posteriors /= posteriors.sum()
        return posteriors


    def _compute_GM_posterior(self, x, priors, GMMs):
        """
        Compute class posteriors, where each class has a GM distribution.
        :param point:
        :param priors: Prior distribution over classes
        :param GMMs: List with each cluster's GMM-density.
        :return:
        """
        posteriors = np.zeros(self.n_clusters)
        for i_cluster in range(self.n_clusters):
            gmm = GMMs[i_cluster]
            density = 0.0
            for i_component in range(gmm.weights_.shape[0]):
                density += gmm.weights_[i_component]*multivariate_normal.pdf(x, mean=gmm.means_[i_component],
                                                                             cov=gmm.covariances_[i_component])
            posteriors[i_cluster] = density*priors[i_cluster]

        posteriors /= posteriors.sum()
        return posteriors

    def _estimate_n_GMM_components(self, x, n_component_lim=[1,4]):

        n_points = x.shape[0]
        n_half = int(n_points / 2.0)
        train_set = x[0:n_half, :]
        val_set = x[n_half + 1::, :]

        min_comp = n_component_lim[0]
        max_comp = n_component_lim[1]

        log_likelihoods = np.zeros(max_comp+1-min_comp)
        counter=0
        for i_comp in range(min_comp,max_comp+1):
             GMM = GaussianMixture(i_comp)
             GMM.fit(train_set)
             log_likelihoods[counter] = GMM.score(val_set)
             counter+=1

        return min_comp + np.argmax(log_likelihoods)

    def _fit_GM(self, proj, n_component_lim=[1,5]):
        """
        Fit a Gaussian mixture model to the data in cluster
        :param proj:
        :return:
        """
        models = []

        for i_cluster in range(self.n_clusters):
            cluster = proj[self.labels == i_cluster, :]

            n_components = self._estimate_n_GMM_components(cluster,n_component_lim)

            GMM = GaussianMixture(n_components)
            GMM.fit(cluster)
            models.append(GMM)

        return models

    def _fit_Gaussians(self,proj):
        """
        Compute mean and covariance of each cluster
        :return:
        """
        means = []
        covs = []

        for i_cluster in range(self.n_clusters):
            cluster = proj[self.labels==i_cluster,:]
            means.append(cluster.mean(axis=0))
            covs.append(np.cov(cluster.T,rowvar=True))

        return means, covs

    def _set_class_prior(self):
        prior = np.zeros(self.n_clusters)

        for i_cluster in range(self.n_clusters):
            prior[i_cluster] = np.sum(self.labels==i_cluster)
        return prior

    def _build_relevance_basis_vector(self,feature_importance, coloring, i_col, j_col):
        """
        Identify a basis vector as the relevances between two colors.
        """

        basis_vector = np.zeros(feature_importance.shape)

        from_residues = np.where(coloring == i_col)[0]
        to_residues = np.where(coloring == j_col)[0]

        n_from = from_residues.shape[0]
        n_to = to_residues.shape[0]

        for i in range(n_from):
            for j in range(n_to):
                basis_vector[from_residues[i],to_residues[j]] = feature_importance[from_residues[i],to_residues[j]]

        basis_vector = squareform(basis_vector + basis_vector.T)
        return basis_vector

    def _identify_relevance_basis_vectors(self, feature_importance):
        """
        Identify the relevance basis vectors for projections using collections of residues.
        """

        # Create binary graph
        graph = np.zeros(feature_importance.shape)
        graph[feature_importance>0] = 1

        # Determine residue coloring
        coloring = nx.coloring.greedy_color(nx.from_numpy_matrix(graph))
        coloring = np.asarray(list(coloring.values()))

        n_colors = int(coloring.max())+1

        relevance_basis_vectors = []

        # Find relevance basis vectors between all color to color combinations
        for i_col in range(n_colors):
            for j_col in range(i_col+1,n_colors):
                tmp_vec = self._build_relevance_basis_vector(feature_importance, coloring, i_col, j_col)
                if np.sum(tmp_vec) > 0:
                    relevance_basis_vectors.append(tmp_vec)

        relevance_basis_vectors = np.asarray(relevance_basis_vectors).T
        if len(relevance_basis_vectors.shape)==1:
            relevance_basis_vectors = relevance_basis_vectors[:,np.newaxis]
        return relevance_basis_vectors

    def _project_on_relevance_basis_vectors(self,distances, relevance_basis_vectors):
        """
        Project all input distances onto the detected basis vectors.
        """

        projected_data = np.dot(distances, relevance_basis_vectors)

        return projected_data

