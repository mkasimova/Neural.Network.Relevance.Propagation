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
from biopandas.pdb import PandasPdb
import modules.utils as utils
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform


logger = logging.getLogger("postprocessing")


class PostProcessor(object):


    def __init__(self, extractor, feature_importance, std_feature_importance, test_set_errors, cluster_indices, working_dir, rescale_results=True, filter_results=False, filter_results_by_cutoff=False, feature_to_resids=None, pdb_file=None, predefined_relevant_residues=None):
        """
        Class which computes all the necessary averages and saves them as fields
        TODO move some functionality from class feature_extractor here
        :param extractor:
        :param feature_importance:
        :param std_feature_importance:
        :param cluster_indices:
        :param working_dir:
        :param feature_to_resids: an array of dimension nfeatures*2 which tells which two residues are involved in a feature
        """
        self.extractor = extractor
        self.feature_importance = feature_importance
        self.std_feature_importance = std_feature_importance
        if rescale_results: #TODO scaling is also performed later when importance per residue is computed; remove this scaling?
            self.feature_importance, self.std_feature_importance = rescale_feature_importance(self.feature_importance, self.std_feature_importance)
        if filter_results:
            self.feature_importance, self.std_feature_importance = filter_feature_importance(self.feature_importance, self.std_feature_importance)
        if filter_results_by_cutoff:
            self.feature_importance, self.std_feature_importance = filter_feature_importance_by_cutoff(self.feature_importance, self.std_feature_importance)

        # Put importance and std to 0 for residues pairs which were filtered out during features filtering (they are set as -1 in self.feature_importance and self.std_feature_importance)
        self.indices_filtered = np.where(self.feature_importance[:,0]==-1)[0]
        self.feature_importance[self.indices_filtered,:] = 0
        self.std_feature_importance[self.indices_filtered,:] = 0

        self.test_set_errors = test_set_errors
        self.cluster_indices = cluster_indices
        self.working_dir = working_dir
        self.pdb_file = pdb_file

        self.nfeatures, self.nclusters = feature_importance.shape
        if feature_to_resids is None and self.pdb_file is None:
            feature_to_resids = utils.get_default_feature_to_resids(self.nfeatures)
        elif feature_to_resids is None and self.pdb_file is not None:
            feature_to_resids = utils.get_feature_to_resids_from_pdb(self.nfeatures,self.pdb_file)
        self.feature_to_resids = feature_to_resids

        self.importance_per_cluster = None
        self.std_importance_per_cluster = None
        self.importance_per_residue_and_cluster = None
        self.std_importance_per_residue_and_cluster = None
        self.importance_per_residue = None
        self.std_importance_per_residue = None
        self.index_to_resid = None

        self.entropy = None
        self.average_std = None

        self.test_set_errors = test_set_errors.mean()

        # Used for toy model
        self.correct_relevance_peaks = None
        self.false_positives = None
        self.predefined_relevant_residues = predefined_relevant_residues

    def average(self):
        """
        Computes average importance per cluster and residue and residue etc.
        Sets the fields importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue
        :return: itself
        """
        self.importance_per_cluster = self.feature_importance # compute_importance_per_cluster(importance, cluster_indices)
        self.std_importance_per_cluster = self.std_feature_importance
        self._compute_importance_per_residue_and_cluster()
        self._compute_importance_per_residue()
        self._compute_entropy()
        self._compute_average_std()

        if self.predefined_relevant_residues is not None:
            self._evaluate_relevance_prediction()

        return self

    def _evaluate_relevance_prediction(self):
        """ 
        Computes number of correct relevance predictions and number of false positives.
        """
        peaks = self._identify_peaks()

        self.correct_relevance_peaks = np.sum(peaks[self.predefined_relevant_residues])
        peaks[self.predefined_relevant_residues] = 0
        self.false_positives = peaks.sum()

        return

    def _identify_peaks(self):
        """ 
        Identifying peaks using k-means clustering. 
        """
        km = KMeans(n_clusters=2).fit(self.importance_per_residue.reshape(-1, 1))
        cluster_indices = km.labels_
        centers = km.cluster_centers_
        peaks = np.zeros(cluster_indices.shape)

        if centers[0] < centers[1]:
            peaks[cluster_indices==1] = 1
        else:
            peaks[cluster_indices==0] = 1

        return peaks

    def persist(self, directory=None):
        """
        Save .npy files of the different averages and pdb files with the beta column set to importance
        :return: itself
        """
        if directory is None:
            directory = self.working_dir + "analysis/{}/".format(self.extractor.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + "importance_per_cluster", self.importance_per_cluster)
        np.save(directory + "importance_per_residue_and_cluster", self.importance_per_residue_and_cluster)
        np.save(directory + "importance_per_residue", self.importance_per_residue)
        np.save(directory + "importance_matrix", self.feature_importance)
        np.save(directory + "std_importance_matrix", self.std_feature_importance)

        if self.pdb_file is not None:
            pdb = PandasPdb()
            pdb.read_pdb(self.pdb_file)
            _save_to_pdb(pdb, directory + "all_importance.pdb", self._map_to_correct_residues(self.importance_per_residue))
            for cluster_idx, importance in enumerate(self.importance_per_residue_and_cluster.T):
                _save_to_pdb(pdb, directory + "cluster_{}_importance.pdb".format(cluster_idx), self._map_to_correct_residues(importance))

        return self

    def _compute_importance_per_residue_and_cluster(self):

        importance = self.importance_per_cluster
        if self.nclusters < 2:
            logger.debug("Not possible to compute importance per cluster")
        index_to_resid = set(self.feature_to_resids.flatten()) # at index X we have residue number
        self.nresidues = len(index_to_resid)
        index_to_resid = [r for r in index_to_resid]
        res_id_to_index = {} # a map pointing back to the index in the array index_to_resid
        for idx, resid in enumerate(index_to_resid):
            res_id_to_index[resid] = idx
        importance_per_residue_and_cluster = np.zeros((self.nresidues, self.nclusters))
        std_importance = np.zeros((self.nresidues, self.nclusters))
        for feature_idx, rel in enumerate(importance):
            res1, res2 = self.feature_to_resids[feature_idx]
            res1 = res_id_to_index[res1]
            res2 = res_id_to_index[res2]
            importance_per_residue_and_cluster[res1, :] += rel
            importance_per_residue_and_cluster[res2, :] += rel
            std_importance[res1,:] += self.std_importance_per_cluster[feature_idx,:]**2
            std_importance[res2,:] += self.std_importance_per_cluster[feature_idx,:]**2
        std_importance = np.sqrt(std_importance)

        importance_per_residue_and_cluster, std_importance = rescale_feature_importance(importance_per_residue_and_cluster, std_importance)

        self.importance_per_residue_and_cluster = importance_per_residue_and_cluster
        self.std_importance_per_residue_and_cluster = std_importance
        self.index_to_resid = index_to_resid

    def _compute_importance_per_residue(self):
        if len(self.importance_per_residue_and_cluster.shape) < 2:
            self.importance_per_residue = self.importance_per_residue_and_cluster
            self.std_importance_per_residue = self.std_importance_per_residue_and_cluster
        else:
            self.importance_per_residue = self.importance_per_residue_and_cluster.mean(axis=1)
            self.std_importance_per_residue = np.sqrt(np.mean(self.std_importance_per_residue_and_cluster**2,axis=1))

    def _map_to_correct_residues(self, importance_per_residue):
        residue_to_importance = {}
        for idx, rel in enumerate(importance_per_residue):
            resSeq = self.index_to_resid[idx]
            residue_to_importance[resSeq] = rel
        return residue_to_importance

    def _compute_entropy(self):
        rel = self.importance_per_residue[self.importance_per_residue > 0]
        rel /= rel.sum()
        self.entropy = -np.sum(rel*np.log(rel))
        return

    def _compute_average_std(self):
        self.average_std = self.std_importance_per_residue.mean()
        return

    def filter_feature_importance_by_rank(self,filter_by_rank_cutoff=None):
        """
        Filter feature importance based on significance
        Keep first 'filter_by_rank_cutoff' ranked features importances
        """
        logger.info("Filtering feature importances by rank %s", filter_by_rank_cutoff)

        n_states = self.feature_importance.shape[1]
        n_features = self.feature_importance.shape[0]

        importance_filt = np.zeros((n_features,n_states))
        std_importance_filt = np.zeros((n_features,n_states))

        for i in range(n_states):
            ind_top_features = np.argsort(-self.feature_importance[:,i])[0:filter_by_rank_cutoff]
            importance_filt[ind_top_features,i] = self.feature_importance[ind_top_features,i]
            std_importance_filt[ind_top_features,i] = self.std_feature_importance[ind_top_features,i]

        self.feature_importance = np.copy(importance_filt)
        self.std_feature_importance = np.copy(std_importance_filt)
        return


def rescale_feature_importance(relevances, std_relevances):
    """
    Min-max rescale feature importances
    :param feature_importance: array of dimension nfeatures * nstates
    :param std_feature_importance: array of dimension nfeatures * nstates
    :return: rescaled versions of the inputs with values between 0 and 1
    """

    logger.info("Rescaling feature importances ...")

    n_states = relevances.shape[1]
    n_features = relevances.shape[0]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:,0]>=0)[0]

    for i in range(n_states):
        max_val, min_val = relevances[indices_not_filtered,i].max(), relevances[indices_not_filtered,i].min()
        scale = max_val-min_val
        offset = min_val
        if scale < 1e-9:
            scale = 1e-9 #TODO correct?
        relevances[indices_not_filtered,i] = (relevances[indices_not_filtered,i] - offset)/scale
        if std_relevances is not None:
            std_relevances[indices_not_filtered, i] /= scale #TODO correct?
    return relevances, std_relevances


def filter_feature_importance(relevances, std_relevances, n_sigma_threshold=2):
    """
    Filter feature importances based on significance
    Return filtered residue feature importances (average + std within the states/clusters)
    """
    logger.info("Filtering feature importances by sigma %s", n_sigma_threshold)

    n_states = relevances.shape[1]
    n_features = relevances.shape[0]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:,0]>=0)[0]

    for i in range(n_states):
        global_mean = np.mean(relevances[indices_not_filtered, i])
        global_sigma = np.std(relevances[indices_not_filtered, i])

        # Identify insignificant features
        ind_below_sigma = np.where(relevances[indices_not_filtered, i] < (global_mean + n_sigma_threshold * global_sigma))[0]
        # Remove insignificant features
        relevances[indices_not_filtered, i][ind_below_sigma] = 0
        std_relevances[indices_not_filtered, i][ind_below_sigma] = 0
    return relevances, std_relevances


def filter_feature_importance_by_cutoff(relevances, std_relevances, cutoff=0.5):
    """
    Filter feature importance based on significance
    Unlike filter_feature_importance uses the same cutoff for all feature extractors
    """
    logger.info("Filtering feature importances by cutoff %s", cutoff)

    n_states = relevances.shape[1]
    n_features = relevances.shape[0]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:,0]>=0)[0]

    for i in range(n_states):
        # Identify insignificant features
        ind_below_cutoff = np.where(relevances[indices_not_filtered, i] <= cutoff)[0]
        # Remove insignificant features
        relevances[indices_not_filtered, i][ind_below_cutoff] = 0
        std_relevances[indices_not_filtered, i][ind_below_cutoff] = 0
    return relevances, std_relevances


def _save_to_pdb(pdb, out_file, residue_to_importance):
    atom = pdb.df['ATOM']
    missing_residues = []
    for i, line in atom.iterrows():
        resSeq = int(line['residue_number'])
        importance = residue_to_importance.get(resSeq, None)
        if importance is None:
            missing_residues.append(resSeq)
            importance = 0
        atom.set_value(i, 'b_factor', importance)
    if len(missing_residues) > 0:
        logger.warn("importance is None for residues %s", set(missing_residues))
    pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)


def residue_importances(feature_importances, std_feature_importances):
    """
    Compute residue importance
    DEPRECATED method ... Here in case we need to merge some of the functionality into the current method
    """
    n_states = feature_importances.shape[0]

    n_residues = squareform(feature_importances[0, :]).shape[0]

    resid_importance = np.zeros((n_states, n_residues))
    std_resid_importance = np.zeros((n_states, n_residues))
    for i_state in range(n_states):
        resid_importance[i_state, :] = np.sum(squareform(feature_importances[i_state, :]), axis=1)
        std_resid_importance[i_state, :] = np.sqrt(np.sum(squareform(std_feature_importances[i_state, :] ** 2), axis=1))
    return resid_importance, std_resid_importance
